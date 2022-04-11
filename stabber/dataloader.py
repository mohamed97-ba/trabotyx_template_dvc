import numpy as np
import os
from os.path import join, split, isdir, isfile, abspath
import torch
from PIL import Image
import random
import collections
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import skimage.io as io
from skimage.transform import rotate, AffineTransform, warp, rescale
from skimage import exposure
from skimage.color import rgb2hsv, hsv2rgb
import tqdm
import PIL


class DatasetTrain(Dataset):

    def __init__(self, root_dir, label_file, resize, inflation, split='train', transform=None, t_transform=None):
        lines = [line.rstrip('\n') for line in open(label_file)]
        self.data_path = [join(root_dir, i + ".txt") for i in lines]

        self.count_p = []
        self.count_n = []

        bar = tqdm.tqdm(self.data_path)
        for i, path in enumerate(bar):
            data = open(path, "r")
            label = data.read(1)
            assert len(label) == 1
            if int(label) == 1:
                self.count_p.append(path)
            else:
                self.count_n.append(path)

        # self.count_p = self.count_p[:400]
        print("Number of negative smaples:", len(self.count_n))
        print("Number of positive samples:", len(self.count_p))
        # random.seed(1996)
        # r_count_p = random.sample(count_p, len(count_n))
        # self.data = r_count_p + count_n
        self.data = self.resample(epoch=0, rand=False)

        self.split = split
        self.transform = transform
        self.t_transform = t_transform
        self.inflate = inflation
        self.dim = resize

    def resample(self, epoch, rand=False):

        random.seed(epoch)
        if rand == True:
            r_count_p = random.sample(self.count_p, len(self.count_n))
        else:
            r_count_p =self.count_p[:len(self.count_n)]
        # self.data = self.count_p[:len(self.count_n)]+self.count_n

        return r_count_p + self.count_n

    # def resample(self):
    #     # random.seed(1996)
    #     # r_count_p = random.sample(self.count_p, len(self.count_n))
    #     num = random.randint(0, 3)
    #     # self.data = self.count_p[:len(self.count_n)]+self.count_n
    #     if (num+1)*len(self.count_n) > len(self.count_p):
    #         return self.count_p[-len(self.count_n):] + self.count_n
    #     else:
    #         return self.count_p[num*len(self.count_n):(num+1)*len(self.count_n)] + self.count_n

    def __getitem__(self, item):

        # assert isfile(self.image_path[item]), self.image_path[item]
        # assert isfile(self.data_path[item]), self.data_path[item]

        s = "_"
        path = os.path.splitext(self.data[item])[0]
        self.image_path = s.join((path.split('_')[0:-4])) + ".jpg"
        self.file_name = path.split('/')[-1]

        image = cv2.imread(self.image_path, -1)

        self.cord = self.file_name.split('_')
        x1, y1, x2, y2 = int(self.cord[-4]), int(self.cord[-3]), int(self.cord[-2]), int(self.cord[-1])

        image[y1:y1+y2, x1:x1+x2, : ] = 0

        # xcenter = x1+x2/2
        # ycenter = y1+y2/2
        # width = self.inflate*x2
        # height = self.inflate*y2
        #
        # distance = max(width, height)
        # x1 = int(xcenter-distance/2)
        # y1 = int(ycenter-distance/2)
        # x2 = int(xcenter+distance/2)
        # y2 = int(ycenter+distance/2)
        #
        # self.height, self.width, _ = image.shape
        #
        # if x1 < 0:
        #     x1 = 0
        # if y1 < 0:
        #     y1 = 0
        # if x2 > self.width:
        #     x2 = self.width
        # if y2 > self.height:
        #     y2 = self.height
        #
        # distance = max((y2-y1), (x2-x1))
        # image = image[y1:y1+distance, x1:x1+distance]
        #
        # cv2.rectangle(image, (x1,y1), ((x1+x2), (y1+y2)), color=(0, 0, 255), thickness=2)

        xcenter = x1+x2/2
        ycenter = y1+y2/2
        width = self.inflate*x2
        height = self.inflate*y2

        x1 = int(xcenter-width/2)
        y1 = int(ycenter-height/2)
        x2 = int(xcenter+width/2)
        y2 = int(ycenter+height/2)

        self.height, self.width, _ = image.shape
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > self.width:
            x2 = self.width
        if y2 > self.height:
            y2 = self.height


        image = image[y1:y2, x1:x2]

        image = cv2.resize(image, dsize=(self.dim, self.dim), interpolation=cv2.INTER_NEAREST)
        # image = cv2.copyMakeBorder(image, 0, int(distance-(y2-y1)), 0, int(distance-(x2-x1)), cv2.BORDER_CONSTANT, value=(0,0,0))
        # image = image[int(y1+((y2-y1)/6)):int(y2-((y2-y1)/6)), int(x1+((x2-x1)/6)):int(x2-((x2-x1)/6))]
        self.new_h, self.new_w, _ = image.shape

        # print(image.shape)
        # cv2.imshow('sd', image)
        # cv2.waitKey(5000)

        data = open(self.data[item], "r")
        label = data.read(1)
        assert len(label) == 1

        if self.split == 'train':

            # # Random crop and resize
            # crop = 10
            # start_x = np.random.randint(low=0, high=crop)
            # end_x = np.random.randint(low=self.new_h-crop, high=self.new_h)
            # start_y = np.random.randint(low=0, high=crop)
            # end_y = np.random.randint(low=self.new_w-crop, high=self.new_w)
            # image = image[start_x:end_x, start_y: end_y]
            # image = cv2.resize(image, (self.new_h, self.new_w))

            # Random Flip
            flip = np.random.randint(low=0, high=2)  # 0 represents flip upside down, 1 represents right to left
            if flip != 0:
                image = cv2.flip(image, flip)

            # Rotation
            ang = np.random.randint(low=-15, high=15)
            image = rotate(image, angle=ang)

            # Blur
            ksize = random.randrange(1, 8, 2)
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)

            # HSV
            # hsv_h = 0.015  # image HSV-Hue augmentation (fraction)
            # hsv_s = 0.7  # image HSV-Saturation augmentation (fraction)
            # hsv_v = 0.4  # image HSV-Value augmentation (fraction)
            # self.augment_hsv(image, hsv_h, hsv_s, hsv_v)

            # brightness and jitter
            # brightness = np.random.randint(low=0, high=3)
            # phi = np.random.randint(low=1, high=4)
            # h, w, c = image.shape
            # noise = np.random.randint(-0.5, 10, (h, w))  # design jitter/noise here
            #
            # if brightness == 1:  # Increase intensity
            #
            #     image = (255 / phi) * (image / (255 / 1)) ** 0.5
            #     image = np.array(image, dtype=np.uint8)
            #     zitter = np.zeros_like(image)
            #     zitter[:, :, 1] = noise
            #     image = cv2.add(image, zitter)
            #
            # elif brightness == 2:  # Decrease intensity
            #
            #     image = (255 / phi) * (image / (255 / 1)) ** 2
            #     image = np.array(image, dtype=np.uint8)
            #     zitter = np.zeros_like(image)
            #     zitter[:, :, 1] = noise
            #     image = cv2.add(image, zitter)



            if self.transform is not None:
                image = self.transform(image)

            image = image.type(torch.float32)
            label = torch.from_numpy(np.asarray(label).astype(np.float16))

            return image, label

        elif self.split == 'val':

            # image = cv2.resize(image, dsize=(self.dim, self.dim), interpolation=cv2.INTER_NEAREST)


            if self.transform is not None:
                image = self.transform(image)

            image = image.type(torch.float32)
            label = torch.from_numpy(np.asarray(label).astype(np.float16))

            return image, label

    def augment_hsv(self, im, hgain=0.5, sgain=0.5, vgain=0.5):
        # HSV color-space augmentation
        if hgain or sgain or vgain:
            r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor((im), cv2.COLOR_BGR2HSV))
            dtype = im.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed

    def __len__(self):
        return len(self.data)

    # def collate_fn(self, batch):
    #     images, mask, objmask, gt_coords = list(zip(*batch))
    #     images = torch.stack([image for image in images])
    #     mask = torch.stack([mask for mask in mask])
    #     objmask = torch.stack([objmask for objmask in objmask])
    #     # gt_coords = torch.stack([gt_coords for gt_coords in gt_coords])
    #     return images, mask, objmask, gt_coords


class DatasetTest(Dataset):

    def __init__(self, root_dir, label_file,resize, inflation, transform=None, t_transform=None):
        lines = [line.rstrip('\n') for line in open(label_file)]
        # print(lines)
        # print(len(lines))
        self.data_path = [join(root_dir, i + ".txt") for i in lines]

        count_p = []
        count_n = []

        bar = tqdm.tqdm(self.data_path)
        for i, path in enumerate(bar):
            data = open(path, "r")
            label = data.read(1)
            assert len(label) == 1
            if int(label) == 1:
                count_p.append(path)
            else:
                count_n.append(path)
        # [: len(count_n)]
        self.data = count_p+count_n

        if not os.path.exists(os.path.join("stem_test")):
            os.makedirs(os.path.join("stem_test"))

        self.split = split
        self.transform = transform
        self.t_transform = t_transform
        self.inflate = inflation
        self.dim = resize

    def __getitem__(self, item):
        # assert isfile(self.image_path[item]), self.image_path[item]

        s = "_"
        path = os.path.splitext(self.data[item])[0]
        self.image_path = s.join((path.split('_')[0:-4])) + ".jpg"
        self.file_name = path.split('/')[-1]

        image_ori = cv2.imread(self.image_path, -1)

        self.cord = self.file_name.split('_')
        x1, y1, x2, y2 = int(self.cord[-4]), int(self.cord[-3]), int(self.cord[-2]), int(self.cord[-1])
        coords = [int(x1), int(y1), int(x1+x2), int(y1+y2)]

        cv2.rectangle(image_ori, (x1, y1), ((x1+x2), (y1+y2)), color=(0, 0, 255), thickness=2)

        # image_ori[y1:y1+y2, x1:x1+x2, : ] = 0

        xcenter = x1+x2/2
        ycenter = y1+y2/2
        width = self.inflate*x2
        height = self.inflate*y2

        x1 = int(xcenter-width/2)
        y1 = int(ycenter-height/2)
        x2 = int(xcenter+width/2)
        y2 = int(ycenter+height/2)

        self.height, self.width, _ = image_ori.shape
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > self.width:
            x2 = self.width
        if y2 > self.height:
            y2 = self.height


        new_cords = [x1,y1,x2,y2]
        image = image_ori[y1:y2, x1:x2]

        # distance = max((y2-y1), (x2-x1))

        # image = cv2.copyMakeBorder(image, 0, int(distance-(y2-y1)), 0, int(distance-(x2-x1)), cv2.BORDER_CONSTANT, value=(0,0,0))
        # cv2.imshow('sdf',image)
        # cv2.waitKey(5000)
        data = open(self.data[item], "r")
        label = data.read(1)
        assert len(label) == 1

        h, w, _ = image.shape

        image = cv2.resize(image, dsize=(self.dim, self.dim), interpolation=cv2.INTER_NEAREST)


        cv_img = image.copy()

        if int(label) == 1:
            color_gt = (0, 255, 0)
            cv2.putText(cv_img, "GT:" + 'STAB', (int(10), int(30)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color=color_gt, thickness=2)
        else:
            color_gt = (0,0,255)
            cv2.putText(cv_img, "GT:" + 'NO-STAB', (int(10), int(30) ), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color=color_gt, thickness=2)

        cv2.imwrite("stem_test/"+os.path.basename(path)+'.jpg', cv_img)


        if self.transform is not None:
            image = self.transform(image)

        image = image
        label = torch.from_numpy(np.asarray(label).astype(np.float16))
        image_ori = torch.from_numpy(image_ori)
        return image, image_ori, label, coords, self.file_name, new_cords
        # return image, label

    def __len__(self):
        return len(self.data)

    # def collate_fn(self, batch):
    #     images, gt_coords, size = list(zip(*batch))
    #     images = torch.stack([image for image in images])
    #
    #     return images, gt_coords, size


def get_loader(root_dir, label_file, resize, inflation, batch_size,
               num_thread=4, pin=False, test=False, split='train', shuffle=True):
    if test is False:
        transform = transforms.Compose([
            # transforms.Resize((400, 400)),#   Not used for current version.
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = DatasetTrain(root_dir, label_file, resize, inflation,
                               transform=transform, t_transform=None, split=split)

    else:
        transform = transforms.Compose([
            # transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = DatasetTest(root_dir, label_file, resize, inflation, transform=transform, t_transform=None)

    if shuffle is True:
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_thread,
                                 pin_memory=pin)
    else:
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_thread,
                                 pin_memory=pin)
    return data_loader


#####################################################################################################################
#####################################################################################################################
class DatasetForward(Dataset):

    def __init__(self, root_dir, resize, inflation, transform=None, t_transform=None):
        self.image_path = [join(root_dir, i) for i in os.listdir(root_dir)]
        self.transform = transform
        self.t_transform = t_transform
        self.inflation = inflation
        self.dim = resize
    def __getitem__(self, item):

        assert isfile(self.image_path[item]), self.image_path[item]
        image = cv2.imread(self.image_path[item], -1)

        # print(self.image_path[item])
        self.height, self.width, _ = image.shape

        image = cv2.resize(image, dsize= (self.dim, self.dim), interpolation=cv2.INTER_NEAREST)

        oldh, oldw, _ = image.shape

        h, w, _ = image.shape
        if self.transform is not None:
            image = self.transform(image)

        image = image.type(torch.float32)

        return image, (self.height, self.width)

    def __len__(self):
        return len(self.image_path)


def get_forward(root_dir, resize, inflation, batch_size, img_size=0, num_thread=4, pin=True, test=False, split='train',
               shuffle=True):
        transform = transforms.Compose([
            # transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = DatasetForward(root_dir, resize, inflation, transform=transform, t_transform=None)
        if shuffle is True:
            data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_thread,
                                     pin_memory=pin)
        else:
            data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_thread,
                                     pin_memory=pin)

        return data_loader
