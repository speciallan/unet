from __future__ import print_function
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import cv2

# 用于语义分割的标记数据，本次实验未使用
Sky = [128, 128, 128]
Building = [128, 0, 0]
Pole = [192, 192, 128]
Road = [128, 64, 128]
Pavement = [60, 40, 222]
Tree = [128, 128, 0]
SignSymbol = [192, 128, 128]
Fence = [64, 64, 128]
Car = [64, 0, 128]
Pedestrian = [64, 64, 0]
Bicyclist = [0, 128, 192]
Unlabelled = [0, 0, 0]

# label颜色字典
COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


# 调整数据，即图片大小和通道数
def adjustData(img, mask, flag_multi_class, num_class):

    # 如果是多分类
    if (flag_multi_class):

        # 转化成灰度图
        img = img / 255

        # 如果是3通道则不变，如果是4通道则变成3通道
        mask = mask[:, :, :, 0] if (len(mask.shape) == 4) else mask[:, :, 0]

        # 先生成全为0的矩阵，为后面多分类结果使用
        new_mask = np.zeros(mask.shape + (num_class,))

        # 对应分类的mask赋值为1
        for i in range(num_class):
            # for one pixel in the image, find the class in mask and convert it into one-hot vector
            # index = np.where(mask == i)
            # index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            # new_mask[index_mask] = 1
            new_mask[mask == i, i] = 1

        # 调整mask的结构
        new_mask = np.reshape(new_mask, (new_mask.shape[0], new_mask.shape[1] * new_mask.shape[2], new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask, (new_mask.shape[0] * new_mask.shape[1], new_mask.shape[2]))
        mask = new_mask

    # 如果img的像素值大于1则归一化在进行分类
    elif (np.max(img) > 1):

        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

    return (img, mask)


# 训练集生成器
# batch_size 训练组大小
# train_path 训练集路径
# image_folder 图像数据路径
# image_color_mode mask_color_mode 是否是灰度图
# image_save_prefix mask_save_prefix 图像保存前缀
# flag_multi_class 是否是多分类
# num_class 多分类数量
# save_to_dir 保存路径
# target_size 输出图像大小
# seed 是否生成相同随机数据
def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="grayscale",
                   mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                   flag_multi_class=False, num_class=2, save_to_dir=None, target_size=(256, 256), seed=1):
    # 能同时产生图片和mask
    # 用相同的seed用于图片和mask生成，确保形变相同
    # 如果要让生成器结果可视化，则设置save_to_dir
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)

    # 通过餐路定义图片生成器
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)

    # 合并成训练集生成器
    train_generator = zip(image_generator, mask_generator)

    # 调整训练集数据中的图像大小
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        yield (img, mask)


# 测试集生成器 测试集数据路径，测试集数量，输入图片大小，是否是多酚类，是否是灰度图
def testGenerator(test_path, num_image=30, target_size=(256, 256), flag_multi_class=False, as_gray=True):
    for i in range(num_image):
        # 之前是使用的skimage.io做图片载入，有一些问题，就改成用opencv实现
        # img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        as_gray_option = as_gray and cv2.IMREAD_GRAYSCALE or cv2.IMREAD_COLOR
        img = cv2.imread(os.path.join(test_path, "%d.png" % i), as_gray_option)

        # 将图像数据变成双通道
        img = img[:, :, 0]

        # 讲像素值变成[0,1]
        img = img / 255

        # resize图片的大小
        # img = trans.resize(img,target_size)
        img = cv2.resize(img, target_size)

        # 如果是多分类，则将数据维度加1
        img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,) + img.shape)
        yield img


# 生成训练numpy文件
def geneTrainNpy(image_path, mask_path, flag_multi_class=False, num_class=2, image_prefix="image", mask_prefix="mask",
                 image_as_gray=True, mask_as_gray=True):
    image_name_arr = glob.glob(os.path.join(image_path, "%s*.png" % image_prefix))
    image_arr = []
    mask_arr = []
    image_as_gray_option = image_as_gray and cv2.IMREAD_GRAYSCALE or cv2.IMREAD_COLOR
    musk_as_gray_option = mask_as_gray and cv2.IMREAD_GRAYSCALE or cv2.IMREAD_COLOR

    # 调整图像数据
    for index, item in enumerate(image_name_arr):
        # img = io.imread(item,as_gray = image_as_gray)
        img = cv2.imread(item, image_as_gray_option)
        img = np.reshape(img, img.shape + (1,)) if image_as_gray else img

        # mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = io.imread(item.replace(image_path, mask_path).replace(image_prefix, mask_prefix), musk_as_gray_option)

        mask = np.reshape(mask, mask.shape + (1,)) if mask_as_gray else mask
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        image_arr.append(img)
        mask_arr.append(mask)

    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr, mask_arr


# 标记可视化
def labelVisualize(num_class, color_dict, img):
    img = img[:, :, 0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]
    return img_out / 255


# 保存输出结果
def saveResult(save_path, npyfile, flag_multi_class=False, num_class=2):
    for i, item in enumerate(npyfile):
        img = labelVisualize(num_class, COLOR_DICT, item) if flag_multi_class else item[:, :, 0]
        # 分割图保存到对应路径
        io.imsave(os.path.join(save_path, "%d_predict.png" % i), img)
        # cv2.imwrite(os.path.join(save_path, "%d_predict.png" % i), img)
