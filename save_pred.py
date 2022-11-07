from os.path import join
import tensorflow as tf
from nets.deeplab import Deeplabv3
from utils.readImage import read_jpg
import cv2
import numpy as np


pred_dir = r'./pred_dir' #预测图片存储路径
test_dir = r'dataset/ImageSets/Segmentation'
img_dir = r'dataset/JPEGImages/'

image_path_list = join(test_dir, 'test.txt') # 'dataset/ImageSets/Segmentation\\test.txt'
print(image_path_list)

test_filenames = open(image_path_list, 'r').read().splitlines() # 'IMG0001_2_2'
test_imgs = [join(img_dir, x) for x in test_filenames] # 'dataset/JPEGImages/IMG0001_2_2'


model =Deeplabv3(input_shape=(640,640,3), classes=5)
model.load_weights(r"weights/ep27-val_loss0.24.h5")

def predict(img):
    feature = tf.cast(img, tf.float32)/127.5 - 1
    # feature = tf.divide(tf.subtract(feature, rgb_mean), rgb_std)
    x = tf.expand_dims(feature, axis=0)
    return model.predict(x)

for i in range(len(test_imgs)):
    input_image = read_jpg(test_imgs[i]+'.jpg') # (640,640,3)，不能rezise因为标签也是这么大
    pred_mask = predict(input_image)  # (1, 640, 640, 5)
    pred_mask = tf.argmax(pred_mask, axis=-1) # TensorShape([1, 640, 640])
    pred_mask = np.array(pred_mask)   
    cv2.imwrite(pred_dir+'/'+test_filenames[i]+'.png',pred_mask[0,:,:])
