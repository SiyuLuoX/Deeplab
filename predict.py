import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from utils.readImage import load_image
from utils.readPath import read_images,get_path

from nets.deeplab import Deeplabv3

'''自定义灰度图调色板'''
from pylab import *
from matplotlib.colors import LinearSegmentedColormap
clist=[(1,0,0),(0,1,0),(0,0,1),(1,0.5,0),(1,0,0.5),(0.5,1,0)]
newcmp = LinearSegmentedColormap.from_list('chaos',clist)


model = Deeplabv3(input_shape=(512, 512, 3), classes=5)
model.load_weights(r"weights/ep27-val_loss0.24.h5")


images = get_path(read_images(root="dataset",train=True),False)
annotations = get_path(read_images(root="dataset",train=True),True)
images.sort(key=lambda x: x.split('/')[-1])
annotations.sort(key=lambda x: x.split('/')[-1])

# np.random.seed(2022)
# 生成index为乱序序列,作为随机下标
index = np.random.permutation(len(images))
images = np.array(images)[index]
anno = np.array(annotations)[index]

# 将原图像和图像分割文件进行组合
dataset = tf.data.Dataset.from_tensor_slices((images, anno))

test = dataset.map(load_image)

# 将数据集进行乱序与分批
test_dataset = test.batch(4)

num = 3
for image, mask in test_dataset.take(1):
    pred_mask = model.predict(image) #(4,320,480,5)
    pred_mask = tf.argmax(pred_mask, axis=-1) #(4,320,480)
    pred_mask = pred_mask[..., tf.newaxis] # (4,320,480,1)
    plt.figure(figsize=(10, 10))
    for i in range(num):
        plt.subplot(num, 3, i*num+1)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(image[i]),cmap=newcmp)
        plt.subplot(num, 3, i*num+2)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(mask[i]),cmap=newcmp)
        plt.subplot(num, 3, i*num+3)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(pred_mask[i]),cmap=newcmp) #(320,480) 取[i,320,480]
        plt.show()