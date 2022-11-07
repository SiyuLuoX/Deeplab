import glob
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt


# 将图像255像素值去掉

def ignore_label(image):
    height = image.shape[0]
    width = image.shape[1]
    # print('width: %s, height: %s, channels: %s'%(width, height))
    # 两层循环逐个修改像素点
    for row in range(height):
        for col in range(width):
            pv = image[row, col]
            if pv == 255:
                image[row, col] = 0


annotations = glob.glob('dataset/SegmentationClass/*.png')

for i in annotations:
    im = Image.open(i)
    im = np.array(im)
    ignore_label(im)
    cv2.imwrite(i,im[:,:])
    print(f"已完成:{i}")

