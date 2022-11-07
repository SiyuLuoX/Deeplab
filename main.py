import datetime
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import (ModelCheckpoint,ReduceLROnPlateau,TensorBoard)
from nets.deeplab import Deeplabv3

from utils.readImage import load_image
from utils.readPath import read_images,get_path


images = get_path(read_images(root="dataset",train=True),False)
annotations = get_path(read_images(root="dataset",train=True),True)
images.sort(key=lambda x: x.split('/')[-1])
annotations.sort(key=lambda x: x.split('/')[-1])

np.random.seed(2022)
# 生成index为乱序序列,作为随机下标
index = np.random.permutation(len(images))
images = np.array(images)[index]
anno = np.array(annotations)[index]

# 将原图像和图像分割文件进行组合
dataset = tf.data.Dataset.from_tensor_slices((images, anno))
# 划分训练数据集和测试数据集
test_count = int(len(images)*0.2)
train_count = len(images) - test_count
dataset_train = dataset.skip(test_count)
dataset_test = dataset.take(test_count)

# 定义批训练有关的参数
BATCH_SIZE = 4
BUFFER_SIZE = 100
STEPS_PER_EPOCH = train_count // BATCH_SIZE
VALIDATION_STEPS = test_count // BATCH_SIZE
WEIGTH_DIR = "./weights"
LOG_DIR = os.path.join('logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))


train = dataset_train.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset_test.map(load_image)

# 将数据集进行乱序与分批
train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)

model = Deeplabv3(input_shape=(512, 512, 3), classes=5)
# 导入预训练模型，冻结部分
weights_path = "deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5"
model.load_weights(weights_path,by_name=True,skip_mismatch=True)
trainable_layer = 130
for i in range(trainable_layer):
    model.layers[i].trainable = False
print('freeze the first {} layers of total {} layers.'.format(trainable_layer, len(model.layers)))

# 训练参数的设置
checkpoint = ModelCheckpoint(os.path.join(WEIGTH_DIR,'ep{epoch:02d}-val_loss{val_loss:.2f}.h5'),
                            monitor='val_accuracy', 
                            save_best_only=True, 
                            save_weights_only=True, # 占用内存小（只保存模型权重）
                            period=3)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)
tensorboard = TensorBoard(log_dir=LOG_DIR)



model.compile(optimizer='adam', 
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

EPOCHS = 50
history = model.fit(train_dataset, 
                    epochs=EPOCHS,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_steps=VALIDATION_STEPS,
                    validation_data=test_dataset,
                    callbacks=[checkpoint, reduce_lr,tensorboard])
