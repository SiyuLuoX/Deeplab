def get_path(filename,suffix=True):
    """ 
    传入文件名返回文件路径

    Args:
        filename:传入文件名
        suffix:默认True,返回png图片路径
    Return:
        返回图片路径
    """
    path = []
    for i in filename:
        if suffix:
            path.append('./dataset/SegmentationClass/'+i+".png")
        else:
            path.append('./dataset/JPEGImages/'+i+".jpg")
    return path 


def read_images(root="dataset",train=True):
    '''
    并返回图片路径,形如:「'2007_000032'」

    Args:
        root:传入「/ImageSets/Segmentation/」的父级路径
        train:是否是训练集
    Return:
        返回文件名列表
    '''
    txt_fname = root + '/ImageSets/Segmentation/' + ('train.txt' if train else 'val.txt')
    with open(txt_fname, 'r') as f:
        filename = f.read().split()
    return filename
