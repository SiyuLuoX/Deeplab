import time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from nets.deeplab import Deeplabv3

'''自定义灰度图调色板'''
from pylab import *
from matplotlib.colors import LinearSegmentedColormap
clist=[(1,0,0),(0,1,0),(0,0,1),(1,0.5,0),(1,0,0.5),(0.5,1,0)]
newcmp = LinearSegmentedColormap.from_list('chaos',clist)


model =Deeplabv3(input_shape=(640,640,3), classes=5)
model.load_weights(r"weights/ep27-val_loss0.24.h5")

image = tf.io.read_file(r'dataset/JPEGImages/IMG0001_0_0.jpg')
image = tf.image.decode_jpeg(image, channels=3)

def predict(img):
    feature = tf.cast(img, tf.float32)/127.5 - 1
    # feature = tf.divide(tf.subtract(feature, rgb_mean), rgb_std)
    x = tf.expand_dims(feature, axis=0)
    return model.predict(x)

T1 = time.perf_counter()
pred_mask = predict(image) #(4,320,480,5)
T2 =time.perf_counter()
print('推理时间{0}'.format((T2-T1)))
pred_mask = tf.argmax(pred_mask, axis=-1) #(4,320,480)
pred_mask = np.array(pred_mask)  
pred_mask = pred_mask.transpose(1,2,0)
# plt.imshow(pred_mask,cmap=newcmp)
plt.imshow(pred_mask)
plt.show()

