import numpy as np
from sklearn.metrics import confusion_matrix
from PIL import Image
from os.path import join


def fast_hist(label, pred, labels=[0,1,2,3,4]):
    '''计算混淆矩阵

    Args:
        label: 转化成一维数组的标签
        pred: 转化成一维数组的预测值
        labels: 类别矩阵
    Returns:
        返回混淆矩阵
	'''
    hist = confusion_matrix(label,pred,labels=labels)
    return hist


def per_class_iu(hist): #分别为每个类别（在这里是19类）计算mIoU，hist的形状(n, n)
    '''
	Args:
        hist: 传入混淆矩阵
    Returns:
        返回每类IoU组成的数组
	'''
    # np.diag(array) 输出矩阵的对角线元素
    # hist.sum(0)=按列相加  hist.sum(1)按行相加
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist)) #矩阵的对角线上的值组成的一维数组/矩阵的所有元素之和，返回值形状(n,)

def pre_class_PA(hist):
    '''
	Args:
        hist: 传入混淆矩阵
    Returns:
        返回每类IoU组成的数组
	'''
    return np.diag(hist) / hist.sum(axis=1)


def compute_mIoU(gt_dir, pred_dir, test_imgs):#计算mIoU的函数
    """Compute IoU and mIoU

    Args:
        gt_dir:Ground Truth 路径
        pred_dir:预测图片所在路径
        test_imgs: 测试集列表目录

    Returns:
        打印输出所有类IoU和mIoU
    """
    labels = [0,1,2,3,4]

    label_info = ['_backgroung_',"plant","mud","sky","conglo"]
    name_classes = np.array(label_info, dtype=np.str) 

    num_classes = 5
    hist = np.zeros((num_classes, num_classes))

    image_path_list = join(test_imgs, 'test.txt') # 在这里打开记录分割图片名称的txt

    gt_imgs = open(image_path_list, 'r').read().splitlines() 
    gt_imgs = [join(gt_dir, x) + '.png' for x in gt_imgs] # 'dataset/SegmentationClass/IMG0001_2_2'
    pred_imgs = open(image_path_list, 'r').read().splitlines() 
    pred_imgs = [join(pred_dir, x) + '.png' for x in pred_imgs] # 'pred_dir\\IMG0001_2_2'

    for ind in range(len(gt_imgs)):
        label = np.array(Image.open(gt_imgs[ind]))
        label = label.reshape(-1)
        pred = np.array(Image.open(pred_imgs[ind]))
        pred = pred.reshape(-1)
        hist += fast_hist(label,pred,labels)
    
    mIoUs = per_class_iu(hist) # 返回验证集图片的逐类别mIoU值组成的数组
    for ind_class in range(num_classes): 
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2))) # round() 方法返回浮点数x的四舍五入值
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))#在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    
    mPA = pre_class_PA(hist)
    print("----------mPA---------")
    for ind_class in range(num_classes): 
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mPA[ind_class] * 100, 2))) # round() 方法返回浮点数x的四舍五入值
    print('===> mPA: ' + str(round(np.nanmean(mPA) * 100, 2)))#在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值

    print('===> PA:{} '.format(np.diag(hist).sum() /hist.sum())) # 计算PA
    return mIoUs

#三个路径分别为 ‘ground truth’,'自己的实验分割结果'，‘分割图片名称txt文件’
compute_mIoU('dataset/SegmentationClass/','pred_dir/','dataset/ImageSets/Segmentation/')