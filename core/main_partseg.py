"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: main_partseg.py
@Time: 2019/12/31 11:17 AM

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
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, MultiStepLR
from data import ShapeNetPart
# from models.curvenet_seg import CurveNet
from models.backbones.apes_seg_backbone import APESSegBackbone
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE
seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]

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
    os.system('cp models/backbones/apes_seg_backbone.py ../checkpoints/'+args.exp_name+'/apes_seg_backbone.py')

def calculate_shape_IoU(pred_np, seg_np, label, class_choice, eva=False):
    label = label.squeeze()
    shape_ious = []
    category = {}
    for shape_idx in range(seg_np.shape[0]):
        if not class_choice:
            start_index = index_start[label[shape_idx]]
            num = seg_num[label[shape_idx]]
            parts = range(start_index, start_index + num)
        else:
            parts = range(seg_num[label[0]])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
        if label[shape_idx] not in category:
            category[label[shape_idx]] = [shape_ious[-1]]
        else:
            category[label[shape_idx]].append(shape_ious[-1])

    if eva:
        return shape_ious, category
    else:
        return shape_ious

def train(args, io):
    writer = SummaryWriter(log_dir="../log")
    train_dataset = ShapeNetPart(partition='trainval', num_points=args.num_points, class_choice=args.class_choice)
    if (len(train_dataset) < 100):
        drop_last = False
    else:
        drop_last = True
    train_loader = DataLoader(train_dataset, num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=drop_last)
    test_loader = DataLoader(ShapeNetPart(partition='test', num_points=args.num_points, class_choice=args.class_choice), 
                            num_workers=8, batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    
    #device = torch.device("cuda" if args.cuda else "cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #io.cprint("Let's use" + str(torch.cuda.device_count()) + "GPUs!")
    io.cprint("Let's use " + str(device))

    seg_num_all = train_loader.dataset.seg_num_all
    seg_start_index = train_loader.dataset.seg_start_index

    # create model
    model = APESSegBackbone().to(device)
    #model = nn.DataParallel(model)
    if(args.model_path!=''):
        model.load_state_dict(torch.load(args.model_path))

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use AdamW")
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-8)
    elif args.scheduler == 'step':
        scheduler = MultiStepLR(opt, [140, 180], gamma=0.1)
    criterion = cal_loss

    best_test_iou = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        train_label_seg = []
        for data, label, seg in tqdm(train_loader):
            seg = seg - seg_start_index
            label_one_hot = np.zeros((label.shape[0], 16))
            for idx in range(label.shape[0]):
                label_one_hot[idx, label[idx]] = 1
            label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
            data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            seg_pred = model(data, label_one_hot)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, seg_num_all), seg.view(-1,1).squeeze())
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
            pred = seg_pred.max(dim=2)[1]               # (batch_size, num_points)
            count += batch_size
            train_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()                  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()       # (batch_size, num_points)
            train_true_cls.append(seg_np.reshape(-1))       # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))      # (batch_size * num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
            train_label_seg.append(label.reshape(-1))
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_label_seg = np.concatenate(train_label_seg)
        train_loss = train_loss*1.0/count
        train_ious = calculate_shape_IoU(train_pred_seg, train_true_seg, train_label_seg, args.class_choice)
        train_iou = np.mean(train_ious)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % (epoch, 
                                                                                                  train_loss,
                                                                                                  train_acc,
                                                                                                  avg_per_class_acc,
                                                                                                  train_iou)
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
        test_label_seg = []
        for data, label, seg in tqdm(test_loader):
            seg = seg - seg_start_index
            label_one_hot = np.zeros((label.shape[0], 16))
            for idx in range(label.shape[0]):
                label_one_hot[idx, label[idx]] = 1
            label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
            data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            seg_pred = model(data, label_one_hot)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, seg_num_all), seg.view(-1,1).squeeze())
            pred = seg_pred.max(dim=2)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            test_true_cls.append(seg_np.reshape(-1))
            test_pred_cls.append(pred_np.reshape(-1))
            test_true_seg.append(seg_np)
            test_pred_seg.append(pred_np)
            test_label_seg.append(label.reshape(-1))
        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        test_label_seg = np.concatenate(test_label_seg)
        test_loss = test_loss*1.0/count
        test_ious = calculate_shape_IoU(test_pred_seg, test_true_seg, test_label_seg, args.class_choice)
        test_iou = np.mean(test_ious)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f, best iou %.6f' % (epoch,
                                                                                              test_loss,
                                                                                              test_acc,
                                                                                              avg_per_class_acc,
                                                                                              test_iou, best_test_iou)
        io.cprint(outstr)
        writer.add_scalars("Loss", {'train': train_loss, 'test': test_loss}, epoch)
        writer.add_scalars("Iou", {'train': train_iou, 'test': test_iou}, epoch)
        if test_iou >= best_test_iou:
            best_test_iou = test_iou
            torch.save(model.state_dict(), '../checkpoints/%s/models/model.pth' % args.exp_name)

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
    classes = ['Airplane_1','Airplane_2','Airplane_3','Airplane_4','Bag_1','Bag_2','Cap_1','Cap_2','Car_1','Car_2','Car_3','Car_4','Chair_1','Chair_2','Chair_3','Chair_4','Earphone_1','Earphone_2','Earphone_3','Guitar_1','Guitar_2','Guitar_3','Knife_1','Knife_2','Lamp_1','Lamp_2','Lamp_3','Lamp_4','Laptop_1','Laptop_2','Motorbike_1','Motorbike_2','Motorbike_3','Motorbike_4','Motorbike_5','Motorbike_6','Mug_1','Mug_2','Pistol_1','Pistol_2','Pistol_3','Rocket_1','Rocket_2','Rocket_3','Skateboard_1','Skateboard_2','Skateboard_3','Table_1','Table_2','Table_3']
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # 创建图形对象和子图布局
    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(2, 2, width_ratios=[1, 0.05], height_ratios=[1, 0.05])
    # 绘制混淆矩阵图像
    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(cm, interpolation='nearest', cmap='plasma')
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
    plt.savefig('../checkpoints/cm_seg', dpi=800)
    plt.show()

def visualize_tsne(data,label,name):
    """
    使用T-SNE算法可视化点云数据
    参数：
    point_cloud: numpy.ndarray，点云数据，维度为 (B, C, N)
    """
    B,N,C = data.shape

    # 将点云数据转换为二维数组 (B*N, C)
    data = data.reshape(B * N, C)
    label = label.reshape(-1)
    # 使用T-SNE算法进行降维
    tsne = TSNE(n_components=2)
    data = tsne.fit_transform(data)
    # 绘制T-SNE图
    plt.scatter(data[:, 0], data[:, 1], c=label)
    plt.title('%s' %name)
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    plt.savefig('../checkpoints/%s' %name, dpi=800)

def test(args, io):
    test_loader = DataLoader(ShapeNetPart(partition='test', num_points=args.num_points, class_choice=args.class_choice),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    #device = torch.device("cuda" if args.cuda else "cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    io.cprint("Let's use " + str(device))

    #Try to load models
    seg_start_index = test_loader.dataset.seg_start_index
    model = APESSegBackbone().to(device)
    #model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path,map_location='cpu'))

    model = model.eval()
    test_acc = 0.0
    test_true_cls = []
    test_pred_cls = []
    test_true_seg = []
    test_pred_seg = []
    test_label_seg = []
    category = {}
    for data, label, seg in tqdm(test_loader):
        # visualize_tsne(data, seg, 'Original point cloud')
        seg = seg - seg_start_index
        label_one_hot = np.zeros((label.shape[0], 16))
        for idx in range(label.shape[0]):
            label_one_hot[idx, label[idx]] = 1
        label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
        data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)

        data = data.permute(0, 2, 1)
        seg_pred = model(data, label_one_hot)

        #ds1
        # ds1_idx = model.backbone.ds1.idx #(16,1024)
        # seg_pred = torch.gather(seg_pred, dim=2, index=ds1_idx.unsqueeze(1).repeat(1, seg_pred.shape[1],1))  # (16,50,1024)
        # seg = torch.gather(seg, dim=1, index=ds1_idx) #(16,1024)

        # ds2
        # ds1_idx = model.backbone.ds1.idx  # (16,1024)
        # ds2_idx = torch.gather(ds1_idx, dim=1, index=model.backbone.ds2.idx)   #(16,512)
        # seg_pred = torch.gather(seg_pred, dim=2, index=ds2_idx.unsqueeze(1).repeat(1, seg_pred.shape[1],1))  # (16,50,512)
        # seg = torch.gather(seg, dim=1, index=ds2_idx) #(16,512)

        #ds3
        # ds1_idx = model.backbone.ds1.idx  # (16,512)
        # ds2_idx = torch.gather(ds1_idx, dim=1, index=model.backbone.ds2.idx)  # (16,512)
        # ds3_idx = torch.gather(ds2_idx, dim=1, index=model.backbone.ds3.idx)   #(16,256)
        # seg_pred = torch.gather(seg_pred, dim=2, index=ds3_idx.unsqueeze(1).repeat(1, seg_pred.shape[1],1))  # (16,50,256)
        # seg = torch.gather(seg, dim=1, index=ds3_idx) #(16,256)

        seg_pred = seg_pred.permute(0, 2, 1).contiguous()
        # visualize_tsne(seg_pred, seg,'Extracted features')
        pred = seg_pred.max(dim=2)[1]
        seg_np = seg.cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        test_true_cls.append(seg_np.reshape(-1))
        test_pred_cls.append(pred_np.reshape(-1))
        test_true_seg.append(seg_np)
        test_pred_seg.append(pred_np)
        test_label_seg.append(label.reshape(-1))

    test_true_cls = np.concatenate(test_true_cls)
    test_pred_cls = np.concatenate(test_pred_cls)
    test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
    test_true_seg = np.concatenate(test_true_seg, axis=0)
    test_pred_seg = np.concatenate(test_pred_seg, axis=0)
    test_label_seg = np.concatenate(test_label_seg)
    test_ious,category = calculate_shape_IoU(test_pred_seg, test_true_seg, test_label_seg, args.class_choice, eva=True)
    if args.cm :
        plot_confusion_matrix(test_true_cls,test_pred_cls)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (test_acc,
                                                                             avg_per_class_acc,
                                                                             np.mean(test_ious))
    io.cprint(outstr)
    results = []
    for key in category.keys():
        results.append((int(key), np.mean(category[key]), len(category[key])))
    results.sort(key=lambda x:x[0])
    for re in results:
        io.cprint('idx: %d mIoU: %.3f num: %d' % (re[0], re[1], re[2]))


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='shapenetpart', metavar='N',
                        choices=['shapenetpart'])
    parser.add_argument('--class_choice', type=str, default=None, metavar='N',
                        choices=['airplane', 'bag', 'cap', 'car', 'chair',
                                 'earphone', 'guitar', 'knife', 'lamp', 'laptop', 
                                 'motor', 'mug', 'pistol', 'rocket', 'skateboard', 'table'])
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
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
    # parser.add_argument('--no_cuda', type=bool, default=False,
    #                     help='enables CUDA training')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',help='Pretrained model path')
    parser.add_argument('--cm', type=bool,  default=False,help='confusion_matrix')
    args = parser.parse_args()

    seed =  1628 #np.random.randint(1, 10000)#1628

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
