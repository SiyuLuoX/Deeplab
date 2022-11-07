import tensorflow as tf
import albumentations as A


#根据图片路径读取一张图片
def read_jpg(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img

#根据图像分割文件路径读取一张图像分割文件
def read_png(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)
    return img

#将输入图片和分割图像文件进行标准化处理
#input_image为待识别的图片，input_mask为分割图像文件
def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32)/127.5 - 1 #使图片每个像素对应的值范围在-1至1之间
    return input_image, input_mask


transform = A.Compose([
    A.RandomCrop(width=512, height=512),
    A.HorizontalFlip(p=0.5), # 水平翻转翻转p为概率
    A.RandomBrightnessContrast(p=0.2),
])


#调用上面三个函数进行图像的读取与处理，返回待识别图像和分割图像文件
def load_image(input_image_path, input_mask_path):
    input_image = read_jpg(input_image_path)
    input_mask = read_png(input_mask_path)

    # transformed = transform(image=input_image,mask=input_mask)
    # img_new=transformed['image']
    # label_new=transformed['mask']

    input_image = tf.image.resize(input_image, (512,512))
    input_mask = tf.image.resize(input_mask, (512,512))
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask