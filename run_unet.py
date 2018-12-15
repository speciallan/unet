#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import sys
import os
import numpy as np
import cv2
import skimage.io as io
from tensorflow.python.keras.callbacks import TensorBoard
# import tensorflow.keras as keras
from spe import *

if device_name == 'cpu':
    from unet.model import *
else:
    from model import *

BACKGROUND = (0, 0, 0)
PAGE = (255, 0, 0)
COLOR_DICT = np.array([BACKGROUND, PAGE])

img_size = (256, 256, 1)

# test = np.array([(0,0,0), (255,0,0), (255,200,200)])
# print(test)
# test = test/255
# print(test)
# test[test > 0.5] = 1
# test[test < 0.5] = 0
# print(test)
# print(test[:,0])
# exit()


# 构造训练集和验证集
data_set, labels, test_set = np.array([np.zeros(img_size)]), np.array([np.zeros(img_size)]), np.array([np.zeros(img_size)])

test_dir = base_dir + 'tf/data/test/math/56/'
output_dir = base_dir + 'tf/data/unet/math/56/'
model_path = base_dir + 'tf/unet/unet_document.hdf5'
# model_path = base_dir + 'tf/unet/unet_membrane.hdf5'

test_imgs = os.listdir(test_dir)
test_imgs.sort()

# 训练集数量
total = num = 54 # 54

# 20/h
epochs = 100

for i, name in enumerate(test_imgs):

    if num <=0:
        break

    if str(name).lower().endswith('.jpg'):

        # img = cv2.imread(dir + name)
        img = cv2.imread(test_dir + name, cv2.IMREAD_GRAYSCALE)
        img_resize = cv2.resize(img, (img_size[0], img_size[1]))
        # spe(img_resize.shape, data_set.shape)

        data_set = np.append(data_set, [img_resize.reshape(img_size)], axis=0)
        # spe(data_set.shape)

        label_dir = test_dir + str(name).replace('.jpg', '').replace('.JPG', '') + '_json/'
        # label_img = cv2.imread(label_dir + 'label.png')
        label_img = cv2.imread(label_dir + 'label.png', cv2.IMREAD_GRAYSCALE)
        label_img_resize = cv2.resize(label_img, (img_size[0], img_size[1]))

        labels = np.append(labels, [label_img_resize.reshape(img_size)], axis=0)

        num -= 1

data_set = np.delete(data_set, 0, axis=0)
labels = np.delete(labels, 0, axis=0)
# spe(data_set.shape, labels.shape)

# labels要转化为标记值
for k,v in enumerate(labels):
    labels = labels / 255
    labels[labels > 0] = 1
    labels[labels <= 0] = 0
    # labels = labels[:,0]

# spe(labels.shape)
# labels = labels[0]
# for i in range(len(labels)):
#     for j in range(len(labels[i])):
#         print(labels[i][j])
# exit()

split = 0.8
train_num = int(total * split)
test_num = 10

X_train, y_train = data_set[:train_num], labels[:train_num]
X_valid, y_valid = data_set[train_num:], labels[train_num:]
# spe(X_valid.shape, y_valid.shape)

X_test = data_set[:test_num]

    # 训练模型
if not os.path.exists(model_path):

    tb = TensorBoard(log_dir='./logs',  # log 目录
                     histogram_freq=1,  # 按照何等频率（epoch）来计算直方图，0为不计算
                     batch_size=1,  # 用多大量的数据计算直方图
                     write_graph=True,  # 是否存储网络结构图
                     write_grads=False,  # 是否可视化梯度直方图
                     write_images=False,  # 是否可视化参数
                     )
    callbacks = [tb]

    model = unet(input_size=img_size)
    model.fit(X_train, y_train, batch_size=1, epochs=epochs, verbose=1, validation_data=(X_valid, y_valid), callbacks=callbacks)
    model.save(model_path)
else:
    model = unet(pretrained_weights=model_path, input_size=img_size)

# 预测
y_pred = model.predict(X_test)
print(y_pred.shape)

def labelVisualize(num_class, color_dict, img):
    img = img[:, :, 0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]
    return img_out / 255

# 保存图片
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i in range(len(y_pred)):
    img_pred = y_pred[i]

    # 如果有预测为1的分类就打印出来
    sum = 0
    for j in range(len(img_pred)):
        for k in range(len(img_pred[j])):
            if img_pred[j][k] == 1:
                sum += img_pred[j][k]
                # print(img_pred[j][k])

    # print('sum=' + str(sum))

    img_pred = labelVisualize(2, COLOR_DICT, img_pred)

    # 这里保存(256,256)
    img_pred = img_pred[:,:,0]
    io.imsave(os.path.join(output_dir, "%d_predict.png" % i), img_pred)
    # cv2.imwrite(os.path.join(output_dir, "%d_predict.png" % i), img_pred)
    print('输出第%d张分割图片成功' %i)

# print(model.summary())
