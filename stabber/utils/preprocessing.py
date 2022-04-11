import glob
import json
import os
import numpy as np
import cv2
import random
import shutil
input_folder = "./dataset/extra-dataset-classification/label"
image_folder = "./dataset/extra-dataset-classification/stab"

output_folder = "./dataset/training"


ann_files = os.listdir(input_folder)

# if not os.path.exists(os.path.join(output_folder+"/data")):
#     os.makedirs(os.path.join(output_folder+"/data"))

# if not os.path.exists(os.path.join("./dataset/extra-dataset-classification"+"/train_set")):
#     os.makedirs(os.path.join("./dataset/extra-dataset-classification"+"/train_set"))

file_train = open(os.path.join("./dataset/extra-dataset-classification", 'train.txt'), "w+")
# file_test = open(os.path.join(output_folder, "test.txt"), "w+")
# file_val = open(os.path.join(output_folder, "val.txt"), "w+")
random.seed(70)
# path = random.sample(ann_files, int(70 * len(ann_files) / 100))
path = os.listdir(image_folder)
# print(path)
# print(ann_files)
train_count_p = 0
train_count_n = 0
test_count_p = 0
test_count_n = 0
postive_sample = []
negative_sample = []
for ann in ann_files:
    # file_r = open(ann)
    if ann in path:
        print(ann)
        continue
    # print(ann)
    file_name = os.path.splitext(ann)[0]
    name = int(open(os.path.join(input_folder+'/' + ann)).read())
    # print(name)

###################################
    # To move from train folder to train_set
    # s = "_"
    # image_path = os.path.join("./dataset/extra-dataset-classification/"+"images/", s.join((ann.split('_')[0:-4])) + ".jpg")
    # if not os.path.exists("./dataset/extra-dataset-classification/"+"train_set/"+ s.join((ann.split('_')[0:-4])) + ".jpg"):
    #     shutil.move(image_path, "./dataset/extra-dataset-classification/"+"train_set/")
####################################

    # print(file_name)
    # s = '_'
    # img_name = os.path.join(image_folder, s.join((file_name.split('_')[0:-4])) + ".jpg")
    # img = cv2.imread(img_name, -1)
    # #
    # coordinates = (file_name.split('/')[-1])
    # cord = coordinates.split('_')
    # x1, y1, x2, y2 = int(cord[-4]), int(cord[-3]), int(cord[-2]), int(cord[-1])

########################################
    # Renaming annotations
    # old_name = os.path.join(input_folder,ann)
    # width = int((x2-x1)/3)
    # x1 = int(x1+((x2-x1)/3))
    # x2 = width
    # height = int((y2-y1)/3)
    # y1 = int(y1+((y2-y1)/3))
    # y2 = height
    # new_cord = [str(x1),str(y1),str(x2),str(y2)]
    # new_name = os.path.join(input_folder,(s.join(file_name.split('_')[0:-4]) + "_" + s.join(new_cord[-4:])+'.txt'))
    # os.rename(old_name, new_name)
##########################################

    # img = img[y1:y1+y2,x1:x1+x2]

    # cv2.imshow("img",img)
    # cv2.waitKey(1000)
    # set = input("Where 1 for train:")
    # cv2.destroyAllWindows()

    # exit()

    # if ann in path:
    #
    # save_name = os.path.join(output_folder + "/data", file_name)
    file_train.write("data/" + file_name + "\n")
    if int(name) == 1:
        train_count_p += 1
    else:
        train_count_n += 1
    #
    # else:
    #
    #     save_name = os.path.join(output_folder+"/data", file_name)
    #     file_test.write("data/" + file_name + "\n")
    #     file_val.write("data/" + file_name + "\n")
    #     if int(name) == 1:
    #         test_count_p += 1
    #     else:
    #         test_count_n += 1


# f.close()
print("Total Stab Class in Training:", train_count_p)
print("Total Don't Stab Class in Training:", train_count_n)
#
# print("Total Stab Class in Testing:", test_count_p)
# print("Total Don't Stab Class in Testing:", test_count_n)
#
# file_test.close()
file_train.close()
# file_val.close()







# postive_sample =[]
# negative_sample = []
# for ann in ann_files:
#     # file_r = open(ann)
#     file_name = os.path.splitext(ann)[0]
#     name = int(open(os.path.join(input_folder+'/' + ann)).read())
#     if name == 1:
#         postive_sample.append(ann)
#     else:
#         negative_sample.append(ann)
# ppath = random.sample(postive_sample,60)
# npath = random.sample(negative_sample,60)




# for ann in ppath:
#     file_name = os.path.splitext(ann)[0]
#     name = int(open(os.path.join(input_folder+'/' + ann)).read())
#     save_name = os.path.join(output_folder+"/data", file_name)
#     file_test.write("data/" + file_name + "\n")
#     file_val.write("data/" + file_name + "\n")
#     if int(name) == 1:
#         test_count_p += 1
#     else:
#         test_count_n += 1
#
# for ann in npath:
#         file_name = os.path.splitext(ann)[0]
#         name = int(open(os.path.join(input_folder + '/' + ann)).read())
#         save_name = os.path.join(output_folder + "/data", file_name)
#         file_test.write("data/" + file_name + "\n")
#         file_val.write("data/" + file_name + "\n")
#         if int(name) == 1:
#             test_count_p += 1
#         else:
#             test_count_n += 1

