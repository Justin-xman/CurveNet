# -*- coding: utf-8 -*-
# @Time : 2023/6/25 09:49
# @Author : Weiliang Zeng
# @Email : weiliang.zeng22@student.edu.cn
# @File : test.py
import torch.nn as nn
import torch
from models.backbones.curvenet_cls import CurveNet
# from mmengine.analysis import get_model_complexity_info
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE


# # 假设你有一个点云特征矩阵 features，形状为 (N, D)，其中 N 是点的数量，D 是特征的维度
# features = np.random.rand(1024, 128)
# # 创建 t-SNE 模型，设置降维后的维度为 2
# tsne = TSNE(n_components=2)
# # 进行 t-SNE 降维
# features_tsne = tsne.fit_transform(features)
# # 提取降维后的坐标
# x = features_tsne[:, 0]
# y = features_tsne[:, 1]
# # 绘制散点图
# plt.scatter(x, y)
# plt.title('t-SNE Visualization')
# # plt.xlabel('Dimension 1')
# # plt.ylabel('Dimension 2')
# plt.show()


# # 绘制混淆图像
# cm = np.random.rand(40, 40)  # 生成一个随机数据
# classes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
# cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# # 创建图形对象和子图布局
# fig = plt.figure(figsize=(15, 15))
# gs = GridSpec(2, 2, width_ratios=[1, 0.05], height_ratios=[1, 0.05])
#
# # 绘制混淆矩阵图像
# ax1 = fig.add_subplot(gs[0, 0])
# im = ax1.imshow(cm, interpolation='nearest', cmap='viridis')
#
# # 添加colorbar
# ax2 = fig.add_subplot(gs[0, 1])
# plt.colorbar(im, cax=ax2)
#
# ax1.set_xticks(np.arange(cm.shape[1]))
# ax1.set_yticks(np.arange(cm.shape[0]))
# ax1.set_xticklabels(classes, fontsize=12)
# ax1.set_yticklabels(classes, fontsize=12)
# ax1.set_title('Normalized Confusion Matrix', fontsize=25)
# ax1.set_xlabel('Predicted label', fontsize=20)
# ax1.set_ylabel('True label', fontsize=20)
#
#     # Rotate the tick labels and set their alignment.
# plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
#
# # 添加标签
# thresh = cm.max() / 2
# for i in range(cm.shape[0]):
#     for j in range(cm.shape[1]):
#         ax1.text(j, i, format(cm[i, j], '.2f'),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
# plt.tight_layout()
# plt.savefig('../checkpoints/cm',dpi=800)
# plt.show()



# net = CurveNet()
# # x = torch.rand(8,3,2048)
# '''方法1，自定义函数 参考自 https://blog.csdn.net/qq_33757398/article/details/109210240'''
# def model_structure(model):
#     blank = ' '
#     print('-' * 90)
#     print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
#           + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
#           + ' ' * 3 + 'number' + ' ' * 3 + '|')
#     print('-' * 90)
#     num_para = 0
#     type_size = 1  # 如果是浮点数就是4
#
#     for index, (key, w_variable) in enumerate(model.named_parameters()):
#         if len(key) <= 30:
#             key = key + (30 - len(key)) * blank
#         shape = str(w_variable.shape)
#         if len(shape) <= 40:
#             shape = shape + (40 - len(shape)) * blank
#         each_para = 1
#         for k in w_variable.shape:
#             each_para *= k
#         num_para += each_para
#         str_num = str(each_para)
#         if len(str_num) <= 10:
#             str_num = str_num + (10 - len(str_num)) * blank
#
#         print('| {} | {} | {} |'.format(key, shape, str_num))
#     print('-' * 90)
#     print('The total number of parameters: ' + str(num_para))
#     print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
#     print('-' * 90)
#
# # model_structure(net)
#
# input_shape = (8,3,2048)
# inputs = torch.rand(8,3,2048)
# analysis_results = get_model_complexity_info(net,input_shape,inputs)
# print(analysis_results['out_table'])

point_cloud = torch.randn(10, 3, 100)  # 替换为你的点云数据张量

# 随机选择 1024 个点的索引
num_points = 10
indices = torch.randperm(100)[:10]

# 使用索引操作从点云数据中选择对应的点，并将索引维度从 (1024,) 调整为 (B, 1024)
sampled_indices = indices.unsqueeze(0).expand(10, -1)
print(indices)