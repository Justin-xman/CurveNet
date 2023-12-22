"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: main_cls.py
@Time: 2018/10/13 10:39 PM

Modified by 
@Author: Tiange Xiang
@Contact: txia7609@uni.sydney.edu.au
@Time: 2021/01/21 3:10 PM
"""

from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from data import ModelNet40
# from models.curvenet_cls import CurveNet
from models.backbones.curvenet_cls import CurveNet
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# from thop import profile
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE

def _init_():
    # fix random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.set_printoptions(10)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

    # prepare file structures
    if not os.path.exists('../checkpoints'):
        os.makedirs('../checkpoints')
    if not os.path.exists('../checkpoints/'+args.exp_name):
        os.makedirs('../checkpoints/'+args.exp_name)
    if not os.path.exists('../checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('../checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp models/utils/layers.py ../checkpoints/'+args.exp_name+'/layers.py')
    os.system('cp models/backbones/curvenet_cls.py ../checkpoints/'+args.exp_name+'/curvenet_cls.py')

def train(args, io):
    writer = SummaryWriter(log_dir="../log")
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    # device = torch.device("cuda" if args.cuda else "cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # io.cprint("Let's use" + str(torch.cuda.device_count()) + "GPUs!")
    io.cprint("Let's use " + str(device))
    
    # create model
    model = CurveNet().to(device)
    # model = nn.DataParallel(model)
    if(args.model_path!=''):
        model.load_state_dict(torch.load(args.model_path))

    # 测试模型的计算量和参数量s
    # input = torch.randn(8,3,2048)
    # flops,params = profile(model,inputs=(data,))
    # # print("参数以及计算量：",flops,params)
    # print("FLOPs: {:.2f}G, Params: {:.2f}M".format(flops / 1e9, params / 1e6))




    if args.use_sgd:
        io.cprint("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        io.cprint("Use AdamW")
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-8)
    elif args.scheduler == 'step':
        scheduler = MultiStepLR(opt, [120, 160], gamma=0.1)
    
    criterion = cal_loss

    best_test_acc = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in tqdm(train_loader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_loss = train_loss*1.0/count
        train_acc = metrics.accuracy_score(train_true, train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f' % (epoch, train_loss,train_acc)
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in tqdm(test_loader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_loss = test_loss*1.0/count
        test_acc = metrics.accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f' % (epoch, test_loss, test_acc)
        io.cprint(outstr)

        writer.add_scalars("Loss", {'train': train_loss, 'test': test_loss}, epoch)
        writer.add_scalars("Acc", {'train': train_acc, 'test': test_acc}, epoch)
        if test_acc >= best_test_acc:
            # best_test_acc = test_acc
            # torch.save(model.state_dict(), '../checkpoints/%s/models/model.pth' % args.exp_name)
            best_test_acc = test_acc
            best_epoch = epoch + 1
            state = {
                'epoch': best_epoch,
                'test_acc': test_acc,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(state, '../checkpoints/%s/models/model.pth' % args.exp_name)
        io.cprint('best: %.3f' % best_test_acc)
        writer.add_scalar('best_test', best_test_acc, epoch)

    writer.flush()
    writer.close()

def plot_confusion_matrix(y_true, y_pred):
    """
    绘制混淆矩阵

    参数：
    y_true: numpy.ndarray，真实标签，维度为 (N,)
    y_pred: numpy.ndarray，预测标签，维度为 (N,)
    classes: list，类别标签列表

    返回值：
    无，直接显示混淆矩阵
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    classes = np.loadtxt('../data/modelnet40_ply_hdf5_2048/shape_names.txt',dtype=str)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # 创建图形对象和子图布局
    fig = plt.figure(figsize=(15, 15))
    gs = GridSpec(2, 2, width_ratios=[1, 0.05], height_ratios=[1, 0.05])
    # 绘制混淆矩阵图像
    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(cm, interpolation='nearest', cmap='viridis')
    # 添加colorbar
    ax2 = fig.add_subplot(gs[0, 1])
    plt.colorbar(im, cax=ax2)
    # We want to show all ticks...
    ax1.set_xticks(np.arange(cm.shape[1]))
    ax1.set_yticks(np.arange(cm.shape[0]))
    ax1.set_xticklabels(classes, fontsize=12)
    ax1.set_yticklabels(classes, fontsize=12)
    ax1.set_title('Normalized Confusion Matrix', fontsize=25)
    ax1.set_xlabel('Predicted label', fontsize=20)
    ax1.set_ylabel('True label', fontsize=20)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 添加标签
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, format(cm[i, j], '.2f'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig('../checkpoints/cm_cls', dpi=800)
    plt.show()

    # # 绘制混淆矩阵
    # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.title('Confusion Matrix')
    # plt.colorbar()
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    # plt.yticks(tick_marks, classes)
    #
    # # 添加标签
    # thresh = cm.max() / 2
    # for i in range(cm.shape[0]):
    #     for j in range(cm.shape[1]):
    #         plt.text(j, i, format(cm[i, j], 'd'),
    #                  horizontalalignment="center",
    #                  color="white" if cm[i, j] > thresh else "black")
    #
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.tight_layout()
    # plt.show()

    #绘制混淆矩阵图像
    # title = 'Normalized confusion matrix'
    # # Compute confusion matrix
    # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    # print(cm)
    # fig = plt.figure(figsize=(18, 18))
    # ax = fig.add_subplot(111)
    # im = ax.imshow(cm, interpolation='nearest', cmap='viridis')#plt.cm.Blues
    # ax.figure.colorbar(im, ax=ax)
    # # We want to show all ticks...
    # ax.set(xticks=np.arange(cm.shape[1]),
    #        yticks=np.arange(cm.shape[0]),
    #        # ... and label them with the respective list entries
    #        xticklabels=classes, yticklabels=classes,
    #        title=title,
    #        ylabel='True label',
    #        xlabel='Predicted label')
    #
    # # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")
    #
    # #Loop over data dimensions and create text annotations.
    # thresh = cm.max() / 2.
    # for i in range(cm.shape[0]):
    #    for j in range(cm.shape[1]):
    #        ax.text(j, i, format(cm[i, j], '.2f'),
    #                ha="center", va="center",
    #                color="white" if cm[i, j] > thresh else "black")
    #
    # fig.tight_layout()
    # fig.show()

def visualize_tsne(data,label,name):
    """
    使用T-SNE算法可视化点云数据
    参数：
    point_cloud: numpy.ndarray，点云数据，维度为 (B, C, N)
    """
    # 使用T-SNE算法进行降维
    tsne = TSNE(n_components=2)
    data = tsne.fit_transform(data)
    # 绘制T-SNE图
    plt.scatter(data[:, 0], data[:, 1], c=label,cmap=plt.get_cmap('hsv'))
    plt.title('%s' %name)
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    plt.savefig('../%s' %name, dpi=800)


def test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    # device = torch.device("cuda" if args.cuda else "cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    io.cprint("Let's use " + str(device))
    #Try to load models
    model = CurveNet().to(device)
    # model = nn.DataParallel(model)
    checkpoint = torch.load(args.model_path,map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label in tqdm(test_loader):

        data, label = data.to(device), label.to(device).squeeze()
        label1 = label.unsqueeze(1).expand(-1, data.shape[1]).reshape(-1) #(B*N)
        B, N, C = data.shape
        data1 = data.reshape(B*N,C)
        # label1 = label1.reshape(-1)
        visualize_tsne(data1, label1, 'Original point cloud')
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        # visualize_tsne(logits.permute(0,2,1).reshape(B*N,-1), label1, 'Extracted features')
        visualize_tsne(logits, label, 'Extracted features')
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    test_acc_avg = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f , test avg acc: %.6f'%(test_acc,test_acc_avg)
    if args.cm :
        plot_confusion_matrix(test_true,test_pred)
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp1', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=30, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    # parser.add_argument('--no_cuda', type=bool, default=True,
    #                     help='enables CUDA training')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=50,
                        help='num of points to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--cm', type=bool,  default=False,help='confusion_matrix')
    args = parser.parse_args()

    seed = 5841 #np.random.randint(1, 10000)

    _init_()

    if args.eval:
        io = IOStream('../checkpoints/' + args.exp_name + '/eval.log')
    else:
        io = IOStream('../checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))
    io.cprint('random seed is: ' + str(seed))
    
    # args.cuda = not args.no_cuda and torch.cuda.is_available()
    #
    # if args.cuda:
    #     io.cprint(
    #         'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
    # else:
    #     io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        with torch.no_grad():
            test(args, io)
