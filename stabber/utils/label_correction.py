import os
import cv2
from pathlib import Path
import shutil



# set = '/home/abhi/PycharmProjects/trabotyx/abhin_ws/stabber/dataset/segregation/training_set_all/reject_class'
# label_set = "/home/abhi/PycharmProjects/trabotyx/abhin_ws/stabber/dataset/test_set/TN/stab"
data_set = '/home/abhi/PycharmProjects/trabotyx/abhin_ws/stabber/dataset/training/data'
label_file = open('/home/abhi/PycharmProjects/trabotyx/abhin_ws/stabber/dataset/training/test.txt')
# val_label_file = open('/home/abhi/PycharmProjects/trabotyx/abhin_ws/stabber/dataset/training/val.txt')
lines = [line.rstrip('\n') for line in (label_file)]
# val_lines = [line.rstrip('\n') for line in os.listdir(data_set)]
lines = [os.path.basename(l) for l in lines]
# files = os.listdir(set)
# files =  [os.path.splitext(f)[0] for f in files]
# label_files = os.listdir(label_set)
# label_files = [os.path.splitext(label)[0] for label in label_files ]
# data_files = [os.path.splitext(x)[0] for x in os.listdir(data_set)]
if not os.path.exists(os.path.join("test_set_agx")):
    os.makedirs(os.path.join("test_set_agx"))

# with open('val_tn.txt', 'w') as file:
for line in lines:

    s = '_'
    img_name = os.path.join(data_set, s.join((line.split('_')[0:-4])) + ".jpg")
    ann_name = os.path.join(data_set, line+'.txt')
    # img = cv2.imread(img_name, -1)
    shutil.copy(img_name, "test_set_agx")
    shutil.copy(ann_name, "test_set_agx")
        # if os.path.basename(line) in files:
        # if line in val_lines:
        #     print(line)
        #     continue
            # gt = data.read(1)

            # if int(gt) == 0:
            #     gt = (gt).replace('0','1')
            # else:
            #     gt = gt.replace('1','0')
            #

            #     print(label_set+'/'+annotation+'.txt')
        # file.write(line+'\n')
# file.close()


            # shutil.copy(data_set+'/'+annotation+'.txt', label_set)