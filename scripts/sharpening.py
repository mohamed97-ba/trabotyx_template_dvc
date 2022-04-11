import numpy as np
import cv2
import os

path = 'set1'
# img = cv2.imread('trab-pwc-0001_1638476551_plugin1.jpg')
original_images = os.listdir(path)

if not os.path.exists('output_sharp'):
    os.makedirs('output_sharp')

def histogram_equalization(img_in):
    # segregate color streams
    b, g, r = cv2.split(img_in)
    h_b, bin_b = np.histogram(b.flatten(), 256, [0, 256])
    h_g, bin_g = np.histogram(g.flatten(), 256, [0, 256])
    h_r, bin_r = np.histogram(r.flatten(), 256, [0, 256])
    # calculate cdf
    cdf_b = np.cumsum(h_b)
    cdf_g = np.cumsum(h_g)
    cdf_r = np.cumsum(h_r)

    # mask all pixels with value=0 and replace it with mean of the pixel values
    cdf_m_b = np.ma.masked_equal(cdf_b, 0)
    cdf_m_b = (cdf_m_b - cdf_m_b.min()) * 255 / (cdf_m_b.max() - cdf_m_b.min())
    cdf_final_b = np.ma.filled(cdf_m_b, 0).astype('uint8')

    cdf_m_g = np.ma.masked_equal(cdf_g, 0)
    cdf_m_g = (cdf_m_g - cdf_m_g.min()) * 255 / (cdf_m_g.max() - cdf_m_g.min())
    cdf_final_g = np.ma.filled(cdf_m_g, 0).astype('uint8')


    cdf_m_r = np.ma.masked_equal(cdf_r, 0)
    cdf_m_r = (cdf_m_r - cdf_m_r.min()) * 255 / (cdf_m_r.max() - cdf_m_r.min())
    cdf_final_r = np.ma.filled(cdf_m_r, 0).astype('uint8')

    # merge the images in the three channels
    img_b = cdf_final_b[b]
    img_g = cdf_final_g[g]
    img_r = cdf_final_r[r]

    img_out = cv2.merge((img_b, img_g, img_r))

    # # validation
    # equ_b = cv2.equalizeHist(b)
    # equ_g = cv2.equalizeHist(g)
    # equ_r = cv2.equalizeHist(r)
    # equ = cv2.merge((equ_b, equ_g, equ_r))
    # # print(equ)
    # # cv2.imwrite('output_name.png', equ)
    return img_out


for f in original_images:

    img = cv2.imread(path+"/"+f)
    img_out = histogram_equalization(img)
    cv2.imwrite("output_sharp/" + f, img_out)
    print(f)
