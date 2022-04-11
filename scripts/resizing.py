import numpy as np
import cv2
import os

path = 'trabotyx_raw'
# img = cv2.imread('trab-pwc-0001_1638476551_plugin1.jpg')
original_images = os.listdir(path)

if not os.path.exists(path+'_resize'):
    os.makedirs(path+'_resize')



for f in original_images:

    img = cv2.imread(path+"/"+f)
    h, w, _ = img.shape
    file_name = os.path.splitext(f)[0]

    if int(h*1.6) < int(w):
        img_1 = img[0:h, 0:int(w/2)]
        img_2 = img[0:h, int(w/2):w]
        img_out_1 = cv2.resize(img_1, (640, 640))
        img_out_2 = cv2.resize(img_2, (640, 640))
        cv2.imwrite(path + '_resize/' + file_name + ".jpg", img_out_1)
        cv2.imwrite(path+'_resize/' + file_name + "_2.jpg", img_out_2)
    elif int(w*1.6) < int(h):
        img_1 = img[0:int(h/2), 0:w]
        img_2 = img[int(h/2):h, 0:w]
        img_out_1 = cv2.resize(img_1, (640, 640))
        img_out_2 = cv2.resize(img_2, (640, 640))
        cv2.imwrite(path + '_resize/' + file_name + ".jpg", img_out_1)
        cv2.imwrite(path+'_resize/' + file_name + "_2.jpg", img_out_2)
    else:
        h = h if h < w else w
        img_1 = img[0:h, 0:h]
        # img_2 = img[0:h, int(w/2):w]
        img_out_1 = cv2.resize(img_1, (640, 640))
        # img_out_2 = cv2.resize(img_2, (640, 640))
        cv2.imwrite(path+'_resize/' + file_name + ".jpg", img_out_1)
        # cv2.imwrite(path+'_resize/' + file_name + "_2.jpg", img_out_2)
    print(f)
    # exit()
