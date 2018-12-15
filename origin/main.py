from data import *
from model import *
import os


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 自己定义的数据字典，用来保存训练集路径、测试集路径、模型保存路径、测试集数量
data_dict = {
    'train_path': '../data/membrane/train',
    'test_path': '../data/membrane/test',
    'model_path': '../unet_membrane.hdf5',
    'test_num': 5
}

# data_dict = {
#     'train_path': 'data/membrane/train',
#     'test_path': 'data/paper/',
#     'model_path': 'unet_membrane.hdf5',
#     'test_num': 4
# }
#
# data_dict = dict(train_path='../data/VOA2007/train',
#     test_path='data/membrane/test',
#     model_path='unet_voa.hdf5',
#     test_num=10)

# 如果没有保存的模型，则训练新模型
if not os.path.exists(data_dict['model_path']):

    # 生成器所需参数
    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')

    # 训练集生成器，包括了路径和参数
    myGene = trainGenerator(2, data_dict['train_path'], 'image', 'label', data_gen_args, save_to_dir=None)

    # 构建unet网络模型
    model = unet()

    # 检测最新的模型参数
    model_checkpoint = ModelCheckpoint(data_dict['model_path'], monitor='loss', verbose=1, save_best_only=True)

    # 训练模型
    model.fit_generator(myGene, steps_per_epoch=300, epochs=1, callbacks=[model_checkpoint])

# 如果有保存的模型，则直接载入模型
else:

    # 载入保存的模型
    model = unet(pretrained_weights=data_dict['model_path'])

# 测试集生成器，用于生成测试集
testGene = testGenerator(data_dict['test_path'], num_image=data_dict['test_num'])

# 预测分割结果
results = model.predict_generator(testGene, data_dict['test_num'], verbose=1)

# 保存预测结果，这里是保存分割图到规定目录
saveResult(data_dict['test_path'], results)
