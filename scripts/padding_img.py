import cv2
import os
import numpy as np

path_main = "set1_padded/"
path_check = "set1/"

if not os.path.exists('set1_raw_padded'):
    os.makedirs('set1_raw_padded')

files = os.listdir(path_main)
files_check = os.listdir(path_check)
for f in files:
    if f in files_check:
        print(f)
        img_main = cv2.imread(path_main+f)
        img = np.mean(img_main, axis=2)
        img_todo = cv2.imread(path_check+f)
        for i in range(3):
                img_todo[:, :, i] = np.where(img == 0, 0, img_todo[:, :, i])

        cv2.imwrite('set1_raw_padded/'+f, img_todo)





# cv2.imshow("img",img_todo)
# cv2.waitKey(0)
# print(np.where(img_main==0))
# print(img_todo)