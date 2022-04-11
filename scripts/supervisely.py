import glob
import json
import os
import numpy as np
import cv2
import random

ann_files = glob.glob("../dataset/ann/*.json")
output_folder = "../dataset/training"


if not os.path.exists(os.path.join(output_folder+"/data")):
    os.makedirs(os.path.join(output_folder+"/data"))

file_train = open(os.path.join(output_folder, 'train.txt'), "w+")
file_test = open(os.path.join(output_folder, "test.txt"), "w+")
file_val = open(os.path.join(output_folder, "val.txt"), "w+")
random.seed(70)
path = random.sample(ann_files, int(70 * len(ann_files) / 100))

for ann in ann_files:
    f = open(ann)
    annotations = json.load(f)



    h = annotations["size"]["height"]
    w = annotations["size"]["width"]
    newh = 100
    neww = 100

    file_name = os.path.splitext(os.path.splitext(os.path.basename(ann))[0])[0]

    # label = np.zeros((newh, neww, 2))
    # mask = np.zeros((newh, neww))
    # rw = neww/w
    # rh = newh/h
    label = np.zeros((h, w, 2))
    mask = np.zeros((h, w))
    # object_mask = np.zeros((h,w))
    rw =1
    rh =1

    gt_coords = {'carrot': [], 'weed': []}

    if not annotations["objects"]:
        data = dict({
            "label": label,
            "coords": gt_coords
        })
        save_name = os.path.join(output_folder + "/data", file_name)
        file_test.write("data/" + file_name + "\n")
        np.save(save_name, data)
        cv2.imwrite(save_name + '_mask.jpg', (mask*255).astype(np.uint8))
        # cv2.imwrite(save_name + '_objmask.jpg', object_mask)
        continue


    for objects in annotations["objects"]:
        y, x = objects["points"]["exterior"][0]  # annotation: (position in width axis, position in height axis)

        y, x = int((y*rh)), int((x*rw))

        class_type = objects["classTitle"]

        if class_type == "carrot":
            gt_coords['carrot'].append((x, y))
            class_id = 1
        else:
            class_id = 2  # weed = 2
            gt_coords['weed'].append((x, y))

        label[x, y, 0] += class_id
        label[x, y, 1] += 1
        mask[x, y] += 1
        # object_mask[x, y] += class_id

    # print(label[x, y, 1])
    label[:, :, 1] = cv2.GaussianBlur(label[:, :, 1], (5, 5), 0)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    # print(label[x, y, 1])
    # exit()
    if mask.max() > 0:
        mask = mask / mask.max()

    if label[:, :, 1].max() > 0:
        label[:, :, 1] = label[:, :, 1] / label[:, :, 1].max()


    gt_coords = (gt_coords)
    data = dict({
        "label": label,
        "coords": gt_coords
    })

    if ann in path:
        save_name = os.path.join(output_folder+"/data", file_name)
        file_train.write("data/" + file_name + "\n")
        np.save(save_name, data)
        cv2.imwrite(save_name + '_mask.jpg', (mask*255).astype(np.uint8))
        # cv2.imwrite(save_name + '_objmask.jpg', object_mask)

    else:
        save_name = os.path.join(output_folder+"/data", file_name)
        file_test.write("data/" + file_name + "\n")
        file_val.write("data/" + file_name + "\n")
        np.save(save_name, data)
        cv2.imwrite(save_name + '_mask.jpg', (mask*255).astype(np.uint8))
        # cv2.imwrite(save_name + '_objmask.jpg', object_mask)

f.close()
file_test.close()
file_train.close()
file_val.close()