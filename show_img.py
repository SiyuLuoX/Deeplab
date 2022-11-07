from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


root="./pred_dir"
fname="/IMG0001_2_2"

# 带坐标和像素值！！好极了
im = plt.imread('%s/%s.png' % (root, fname))
plt.imshow(im)
plt.show()
