import cv2
import os
import numpy as np
import random
from skimage.measure import label, regionprops

weed_path = "/home/abhi/PycharmProjects/trabotyx/abhin_ws/dataset/weeds/"
carrot_path = "/home/abhi/PycharmProjects/trabotyx/abhin_ws/dataset/carrots/"
img_path = "/home/abhi/PycharmProjects/trabotyx/abhin_ws/dataset/images/"

output_folder = "../dataset/training"

if not os.path.exists(os.path.join(output_folder+"/syn_data")):
    os.makedirs(os.path.join(output_folder+"/syn_data"))

file_train = open(os.path.join(output_folder, 'syn_train.txt'), "w+")
file_test = open(os.path.join(output_folder, "syn_test.txt"), "w+")
file_val = open(os.path.join(output_folder, "syn_val.txt"), "w+")

ann_files = os.listdir(img_path)
random.seed(85)
path = random.sample(ann_files, int(85 * len(ann_files) / 100))

# Boundary pixels to ignore. Synthetic annotation also provides positions which are not completely visible
boundary = 5

for ann in ann_files:
    file_name = os.path.splitext(os.path.splitext(os.path.basename(ann))[0])[0]
    print(file_name)
    weed = cv2.imread(weed_path+file_name+'weeds.jpg', 0)
    carrot = cv2.imread(carrot_path+file_name+"carrots.jpg", 0)
    image = cv2.imread(img_path+ann, -1)

    h, w, _ = image.shape
    annotation = np.zeros((h, w, 2))
    mask = np.zeros((h, w))

    gt_coords = {'carrot': [], 'weed': []}

    binary_kmap_w = weed > 200
    binary_kmap_c = carrot > 200
    binary_mask = np.zeros((h, w))

    kmap_label_w = label(binary_kmap_w, connectivity=1)
    kmap_label_c = label(binary_kmap_c, connectivity=1)

    props = regionprops(kmap_label_c)
    plist = []
    for prop in props:
        x, y = np.round(prop.centroid).astype(np.int32)
        if boundary < x < h-boundary and boundary < y < w-boundary:

            gt_coords['carrot'].append((x, y))
            class_id = 1
            annotation[x, y, 0] += class_id
            annotation[x, y, 1] += 1
            mask[x, y] += 1
            # visualise
            image[x-2:x+2, y-2:y+2, :] = [1, 1, 255]

    props = regionprops(kmap_label_w)
    plist = []
    for prop in props:
        x, y = np.round(prop.centroid).astype(np.int32)
        if boundary < x < h - boundary and boundary < y < w - boundary:

            gt_coords['weed'].append((x, y))
            class_id = 2  # weed = 2
            annotation[x, y, 0] += class_id
            annotation[x, y, 1] += 1
            mask[x, y] += 1
            # visualise
            image[x-2:x+2, y-2:y+2, :] = [255, 1, 1]

    annotation[:, :, 1] = cv2.GaussianBlur(annotation[:, :, 1], (5, 5), 0)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    if mask.max() > 0:
        mask = mask / mask.max()

    if annotation[:, :, 1].max() > 0:
        annotation[:, :, 1] = annotation[:, :, 1] / annotation[:, :, 1].max()

    gt_coords = gt_coords
    data = dict({
        "label": annotation,
        "coords": gt_coords
    })

    if ann in path:
        save_name = os.path.join(output_folder+"/syn_data", file_name)
        file_train.write("syn_data/" + file_name + "\n")
        np.save(save_name, data)
        cv2.imwrite(save_name + '_mask.jpg', (mask*255).astype(np.uint8))

    else:
        save_name = os.path.join(output_folder+"/syn_data", file_name)
        file_test.write("syn_data/" + file_name + "\n")
        file_val.write("syn_data/" + file_name + "\n")
        np.save(save_name, data)
        cv2.imwrite(save_name + '_mask.jpg', (mask*255).astype(np.uint8))


file_test.close()
file_train.close()
file_val.close()