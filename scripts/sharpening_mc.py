import cv2
import numpy as np
import os
# img_path = 'trab-pwc-0001_1638476551_plugin1.jpg'
#Reading image

path = 'set1'
# img = cv2.imread('trab-pwc-0001_1638476551_plugin1.jpg')
original_images = os.listdir(path)

if not os.path.exists('output_sharp_mc'):
    os.makedirs('output_sharp_mc')



for f in original_images:

    img = cv2.imread(path+"/"+f)
    print(f)

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # convert from RGB color-space to YCrCb
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # equalize the histogram of the Y channel
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

    # convert back to RGB color-space from YCrCb
    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

    # Sharpenning
    # Define the sharpenning filter
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    # Applying conv operation
    image_sharp = cv2.filter2D(equalized_img, ddepth=-1, kernel=kernel)
    # cv2.imwrite('weed_img_sharp.jpg', equalized_img)
    cv2.imwrite("output_sharp_mc/" + f, equalized_img)
    # Showing results

    # cv2.imshow('Histogram equalized', equalized_img)
    # cv2.imshow('edges', edges)
    # cv2.imshow('sharp', image_sharp)
    # cv2.imshow('laplacien Edge Detection', lap)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)