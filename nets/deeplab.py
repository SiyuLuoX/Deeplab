import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Activation,BatchNormalization,Concatenate,Conv2D,
                                    DepthwiseConv2D,Dropout,GlobalAveragePooling2D,
                                    Input,Lambda,Softmax,ZeroPadding2D)
from tensorflow.keras.models import Model

# from keras import backend as K
# from keras.layers import (Activation, BatchNormalization, Concatenate, Conv2D,
#                           DepthwiseConv2D, Dropout, GlobalAveragePooling2D,
#                           Input, Lambda, Softmax, ZeroPadding2D)
# from keras.models import Model

from nets.mobilenetV2 import mobilenetV2

def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'
    
    if not depth_activation:
        x = Activation('relu')(x)

    # 首先使用3x3的深度可分离卷积
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    # 利用1x1卷积进行通道数调整
    x = Conv2D(filters, (1, 1), padding='same', use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    return x

def Deeplabv3(input_shape=(512, 512, 3), classes=21, alpha=1.):
    img_input = Input(shape=input_shape)

    # x         32,32,320
    # skip1     128,128,24
    x, skip1 = mobilenetV2(img_input, alpha)
    atrous_rates = (6,12,18)
    size_before = tf.keras.backend.int_shape(x)

    #   调整通道 32,32,256
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation('relu', name='aspp0_activation')(b0)
    
    
    # 分支1 rate = 6 (12)   32,32,320->32,32,256
    b1 = SepConv_BN(x, 256, 'aspp1',
                    rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
    # 分支2 rate = 12 (24)  32,32,320->32,32,256
    b2 = SepConv_BN(x, 256, 'aspp2',
                    rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
    # 分支3 rate = 18 (36)  32,32,320->32,32,256
    b3 = SepConv_BN(x, 256, 'aspp3',
                    rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)
    
    #   全部求平均后，再利用expand_dims扩充维度
    #   32,32,320 -> 1,1,256
    b4 = GlobalAveragePooling2D()(x)  # B,H,W,C ->B,C
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation('relu')(b4)
    # 1,1,256 -> 32,32,256
    b4 = Lambda(lambda x: tf.image.resize(x, size_before[1:3]))(b4)

    # 32,32,256 + 32,32,256 -> 32, 32, 512
    x = Concatenate()([b4, b0, b1, b2, b3])

    # 利用1x1卷积调整通道数
    # 32, 32, 1280 -> 32,32,256
    x = Conv2D(256, (1, 1), padding='same', use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    # 将加强特征边上采样
    # 32,32,256 -> 128,128,256
    skip_size = tf.keras.backend.int_shape(skip1)
    x = Lambda(lambda xx: tf.image.resize(xx, skip_size[1:3], method="bilinear"))(x)

    # 浅层特征边
    # 128,128,24 -> 128,128,48
    dec_skip1 = Conv2D(48, (1, 1), padding='same',use_bias=False, name='feature_projection0')(skip1)
    dec_skip1 = BatchNormalization(name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
    dec_skip1 = Activation(tf.nn.relu)(dec_skip1)

    # 与浅层特征堆叠
    # 128,128,48 + 128,128,256 =128,128,xxx
    # 128,128,256
    x = Concatenate()([x, dec_skip1])
    x = SepConv_BN(x, 256, 'decoder_conv0',
                    depth_activation=True, epsilon=1e-5)
    x = SepConv_BN(x, 256, 'decoder_conv1',
                    depth_activation=True, epsilon=1e-5)


    # 128,128,256 -> 128,128,21 -> 512,512,21
    size_before3 = tf.keras.backend.int_shape(img_input)
    x = Conv2D(classes, (1, 1), padding='same')(x)
    x = Lambda(lambda xx:tf.image.resize(xx, size_before3[1:3]))(x)

    x = Softmax()(x)

    model = Model(img_input, x, name='deeplabv3plus')
    return model
