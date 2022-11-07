from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from nets.deeplab import Deeplabv3

from utils.readImage import read_jpg


# root="./dataset"
# fname="/SegmentationClass/DJI_0001_0_0"

# 带坐标和像素值！！好极了
# image = plt.imread("D:\DataSet\BigDataset\JPEGImages\IMG0001.jpg")
# image = np.array(image)

# 砂岩、泥岩、植物，天空，砾岩
colormap = [[243, 156, 18],[146, 144, 143],[39, 174, 96],[125, 60, 152],[231, 76, 60]] # 黄、灰、绿、紫、红

# input_image_path = r'D:/DataSet/BigDataset/JPEGImages/IMG1789.jpg'
input_image_path = r'D:/RAW/9.15erdie/DJI_0878.jpg'
input_image = read_jpg(input_image_path)
input_image = tf.image.resize(input_image, (2240, 4000))

model = Deeplabv3(input_shape=(2240, 4000, 3), classes=5)
model.load_weights(r"weights/ep27-val_loss0.24.h5")

def predict(img):
    feature = tf.cast(img, tf.float32)/127.5 - 1
    # feature = tf.divide(tf.subtract(feature, rgb_mean), rgb_std)
    x = tf.expand_dims(feature, axis=0)
    return model.predict(x)

pred_mask = predict(input_image) #(1, 2240, 4000, 5)
pred_mask = tf.argmax(pred_mask, axis=-1) #TensorShape([1, 2240, 4000])
pred_mask = np.array(pred_mask)  
pred_mask = pred_mask.transpose(1,2,0)

nh = pred_mask.shape[0]  #2240
nw = pred_mask.shape[1]

pred_mask = np.reshape(pred_mask,(nh,nw))

# 创建一副新图，并根据每个像素点的种类赋予颜色

seg_img = np.zeros((nh, nw,3))
for c in range(5):
    seg_img[:,:,0] += ((pred_mask[:,:] == c )*(colormap[c][0])).astype('uint8')
    seg_img[:,:,1] += ((pred_mask[:,:] == c )*(colormap[c][1])).astype('uint8')
    seg_img[:,:,2] += ((pred_mask[:,:] == c )*(colormap[c][2])).astype('uint8')
seg_img = Image.fromarray(np.uint8(seg_img))
plt.imshow(seg_img)
plt.show()




# pred_mask = pred_mask[..., tf.newaxis] # (1, 2240, 4000, 1)

# plt.subplot(1, 2, 1)
# plt.imshow(tf.keras.preprocessing.image.array_to_img(input_image))
# plt.subplot(1, 2, 2)
# plt.imshow(tf.keras.preprocessing.image.array_to_img(pred_mask[0]))
# plt.show()