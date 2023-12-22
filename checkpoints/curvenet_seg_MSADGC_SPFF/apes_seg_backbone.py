from ..utils.layers import Embedding, N2PAttention,down,down_feature,SA_Layer,Attention#, GlobalDownSample, LocalDownSample, UpSample
import torch
from torch import nn
from einops import reduce, pack, repeat


class backbone(nn.Module):
    def __init__(self):
        super(backbone, self).__init__()
        self.embedding = Embedding()
        # self.ds1 = down(1024)  # 2048 pts -> 1024 pts
        # # self.ds2 = down(512)  # 1024 pts -> 512 pts
        # self.ds2 = down_feature(512)
        self.ds1 = down_feature(1024)
        self.ds2 = down(512)
        # self.n2p_attention1 = N2PAttention()
        # self.n2p_attention2 = N2PAttention()
        self.sa1 = SA_Layer(128)
        self.sa2 = SA_Layer(128)
        # self.sa2 = Attention(128)

        #两次采样
        self.conv = nn.Sequential(nn.Conv1d(128, 1024, 1, bias=False), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2))
        self.conv1 = nn.Sequential(nn.Conv1d(128, 512, 1, bias=False), nn.BatchNorm1d(512), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(128, 512, 1, bias=False), nn.BatchNorm1d(512), nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv1d(16, 64, 1, bias=False), nn.BatchNorm1d(64), nn.LeakyReLU(0.2))


    def forward(self, x, shape_class):
        batch_size = x.size(0)
        # 标准版本
        # x_list = []
        # y1 = self.embedding(x)  # (B, 3, 2048) -> (B, 128, 2048)
        # # y1 = self.n2p_attention1(y1)  # (B, 128, 2048) -> (B, 128, 2048)
        # y1 = self.sa1(y1)
        # x_list.append(self.conv(y1).max(dim=-1)[0])  # (B, 128, 2048) -> (B, 1024, 2048) -> (B, 1024)
        # x2, y2 = self.ds1(x, y1)  # (B, 128, 2048) -> (B, 128, 1024)
        # # y2 = self.n2p_attention2(y2)  # (B, 128, 2048) -> (B, 128, 2048)
        # y2 = self.sa2(y2)
        # x_list.append(self.conv1(y2).max(dim=-1)[0])  # (B, 128, 1024) -> (B, 512, 1024) -> (B, 512)
        # # #两次采样
        # y2 = self.conv2(y2)  # (B, 128, 1024) -> (B, 512, 1024)
        # x3, y3 = self.ds2(x2, y2)  # (B, 512, 1024) -> (B, 512, 512)
        # x_list.append(y3.max(dim=-1)[0])  # (B, 512, 512) -> (B, 512)

        #交换采样顺序（先特征再曲率）
        # x_list = []
        # y1 = self.embedding(x)  # (B, 3, 2048) -> (B, 128, 2048)
        # y1 = self.sa1(y1)       # (B, 128, 2048) -> (B, 128, 2048)
        # x_list.append(self.conv(y1).max(dim=-1)[0])  # (B, 128, 2048) -> (B, 1024, 2048) -> (B, 1024)
        # y2 = self.conv2(y1)  # (B, 128, 2048) -> (B, 512, 2048)
        # x2, y2 = self.ds1(x, y2)  # (B, 512, 2048) -> (B, 512, 1024)
        # x_list.append(y2.max(dim=-1)[0])  # (B, 512, 1024) -> (B, 512)
        # x3, y3 = self.ds2(x2, y1)  # (B, 128, 1024) -> (B, 128, 512)
        # y3 = self.sa2(y3)         # (B, 128, 512) -> (B, 128, 512)
        # x_list.append(self.conv1(y3).max(dim=-1)[0])  # (B, 128, 512) -> (B, 512, 512) -> (B, 512)

        #新采样
        x_list = []
        y1 = self.embedding(x)  # (B, 3, 2048) -> (B, 128, 2048)
        # y1 = self.n2p_attention1(y1)        # (B, 128, 2048) -> (B, 128, 2048)
        x_list.append(self.conv(y1).max(dim=-1)[0])  # (B, 128, 2048) -> (B, 1024, 2048) -> (B, 1024)
        y1 = self.sa1(y1)   # (B, 128, 2048) -> (B, 128, 2048)
        x2, y2 = self.ds1(x, y1)  # (B, 128, 2048) -> (B, 128, 1024)
        # y2 = self.n2p_attention2(y2)    # (B, 128, 1024) -> (B, 128, 1024)
        x_list.append(self.conv1(y2).max(dim=-1)[0])  # (B, 128, 1024) -> (B, 512, 1024) -> (B, 512)
        y2 = self.sa2(y2)   # (B, 128, 2048) -> (B, 128, 2048)
        x3, y3 = self.ds2(x2, y2)  # (B, 128, 1024) -> (B, 128, 512)
        x_list.append(self.conv2(y3).max(dim=-1)[0])  # (B, 128, 512) -> (B, 512, 512) -> (B, 512)
        z = torch.cat(x_list, dim=1)  # (B, 2048)
        shape_class = shape_class.unsqueeze(dim=2)
        shape_class = self.conv4(shape_class)  # (B, 16, 1) -> (B, 64, 1)
        y, _ = pack([z, shape_class], 'B *')  # (B, 2112)
        y = repeat(y, 'B C -> B C N', N=1024)  # (B, 2112) -> (B, 2112, 2048)
        x = torch.cat([y1, y], dim=1)  # (B, C=2240, 2048)

        # 只需要特征层MSADGC
        # y = self.embedding(x)  # (B, 3, 2048) -> (B, 128, 2048)
        # y = self.sa1(y)         # (B, 128, 2048) -> (B, 128, 2048)
        # y1 = self.conv(y).max(dim=-1)[0] # (B, 128, 2048) -> (B, 1024, 2048) -> (B, 1024)
        # shape_class = shape_class.unsqueeze(dim=2)
        # shape_class = self.conv4(shape_class)  # (B, 16, 1) -> (B, 64, 1)
        # # shape_class = shape_class.view(batch_size,64)
        # # z = torch.cat((y1,shape_class),dim=1)    #(B, 1088)
        # z, _ = pack([y1, shape_class], 'B *')  # (B, 1088)
        # z = repeat(z, 'B C -> B C N', N=2048)  # (B, 1088) -> (B, 1088, 2048)
        # x = torch.cat([y, z], dim=1)  # (B, C=1216, 2048)
        return y1

class head(nn.Module):
    def __init__(self):
        super(head, self).__init__()
        # self.conv1 = nn.Sequential(nn.Conv1d(2176, 256, 1, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.2))
        # self.conv1 = nn.Sequential(nn.Conv1d(1216, 256, 1, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.2))
        self.conv1 = nn.Sequential(nn.Conv1d(2240, 256, 1, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.2))
        # self.conv1 = nn.Sequential(nn.Conv1d(2496, 256, 1, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(256, 256, 1, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(256, 128, 1, bias=False), nn.BatchNorm1d(128), nn.LeakyReLU(0.2))
        self.conv4 = nn.Conv1d(128, 50, 1, bias=False)
        self.dp1 = nn.Dropout(p=0.5)
        self.dp2 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x)  # (B, 2240, 2048) -> (B, 256, 2048)
        x = self.dp1(x)  # (B, 256, 2048) -> (B, 256, 2048)
        x = self.conv2(x)  # (B, 256, 2048) -> (B, 256, 2048)
        x = self.dp2(x)  # (B, 256, 2048) -> (B, 256, 2048)
        x = self.conv3(x)  # (B, 256, 2048) -> (B, 128, 2048)
        x = self.conv4(x)  # (B, 128, 2048) -> (B, 50, 2048)
        return x

class APESSegBackbone(nn.Module):
    def __init__(self):
        super(APESSegBackbone, self).__init__()
        # self.embedding = Embedding()
        # self.ds1 = down(1024)  # 2048 pts -> 1024 pts
        # self.ds2 = down(512)  # 1024 pts -> 512 pts
        # self.ds2 = down_feature(512)
        # self.n2p_attention1 = N2PAttention()
        # self.n2p_attention2 = N2PAttention()
        # self.n2p_attention3 = N2PAttention()
        # self.n2p_attention4 = N2PAttention()
        # self.n2p_attention5 = N2PAttention()
        # self.ups1 = UpSample()
        # self.ups2 = UpSample()
        # self.conv1 = nn.Sequential(nn.Conv1d(128, 1024, 1, bias=False), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2))
        # self.conv2 = nn.Sequential(nn.Conv1d(16, 64, 1, bias=False), nn.BatchNorm1d(64), nn.LeakyReLU(0.2))
        # self.conv = nn.Sequential(nn.Conv1d(128, 1024, 1, bias=False), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2))
        # self.conv1 = nn.Sequential(nn.Conv1d(128, 512, 1, bias=False), nn.BatchNorm1d(512), nn.LeakyReLU(0.2))
        # self.conv2 = nn.Sequential(nn.Conv1d(128, 512, 1, bias=False), nn.BatchNorm1d(512), nn.LeakyReLU(0.2))
        # self.conv3 = nn.Sequential(nn.Conv1d(16, 64, 1, bias=False), nn.BatchNorm1d(64), nn.LeakyReLU(0.2))
        # self.conv4 = nn.Sequential(nn.Conv1d(2240, 256, 1, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.2))
        # self.conv5 = nn.Sequential(nn.Conv1d(256, 256, 1, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.2))
        # self.conv6 = nn.Sequential(nn.Conv1d(256, 128, 1, bias=False), nn.BatchNorm1d(128), nn.LeakyReLU(0.2))
        # self.conv7 = nn.Conv1d(128, 50, 1, bias=False)
        # self.dp1 = nn.Dropout(p=0.5)
        # self.dp2 = nn.Dropout(p=0.5)
        self.backbone = backbone()
        self.head = head()

    def forward(self, x, shape_class):
        # tmp = self.embedding(x)  # (B, 3, 2048) -> (B, 128, 2048)
        # x1 = self.n2p_attention1(tmp)  # (B, 128, 2048) -> (B, 128, 2048)
        # tmp = self.ds1(x1)  # (B, 128, 2048) -> (B, 128, 1024)
        # x2 = self.n2p_attention2(tmp)  # (B, 128, 1024) -> (B, 128, 1024)
        # tmp = self.ds2(x2)  # (B, 128, 1024) -> (B, 128, 512)
        # x3 = self.n2p_attention3(tmp)  # (B, 128, 512) -> (B, 128, 512)
        # tmp = self.ups2(x2, x3)  # (B, 128, 512) -> (B, 128, 1024)
        # x2 = self.n2p_attention4(tmp)  # (B, 128, 1024) -> (B, 128, 1024)
        # tmp = self.ups1(x1, x2)  # (B, 128, 1024) -> (B, 128, 2048)
        # x1 = self.n2p_attention5(tmp)  # (B, 128, 2048) -> (B, 128, 2048)
        # x2 = self.ds1(x1)  # (B, 128, 2048) -> (B, 128, 1024)
        # x3 = self.ds2(x2)  # (B, 128, 1024) -> (B, 128, 512)
        # x2 = self.ups2(x2, x3)  # (B, 128, 512) -> (B, 128, 1024)
        # x1 = self.ups1(x1, x2)  # (B, 128, 1024) -> (B, 128, 2048)
        # x = self.conv1(x1)  # (B, 128, 2048) -> (B, 1024, 2048)
        # x_max = reduce(x, 'B C N -> B C', 'max')  # (B, 1024, 2048) -> (B, 1024)
        # x_avg = reduce(x, 'B C N -> B C', 'mean')  # (B, 1024, 2048) -> (B, 1024)
        # x, _ = pack([x_max, x_avg], 'B *')  # (B, 1024) -> (B, 2048)
        # shape_class = self.conv2(shape_class)  # (B, 16, 1) -> (B, 64, 1)
        # x, _ = pack([x, shape_class], 'B *')  # (B, 2048) -> (B, 2112)
        # x = repeat(x, 'B C -> B C N', N=2048)  # (B, 2112) -> (B, 2112, 2048)
        # x, _ = pack([x, x1], 'B * N')  # (B, 2112, 2048) -> (B, 2240, 2048)

        #连续降采样
        # x_list = []
        # y1 = self.embedding(x)  # (B, 3, 2048) -> (B, 128, 2048)
        # y1 = self.n2p_attention1(y1)  # (B, 128, 2048) -> (B, 128, 2048)
        # x_list.append(self.conv(y1).max(dim=-1)[0])  # (B, 128, 2048) -> (B, 1024, 2048) -> (B, 1024)
        # x2,y2 = self.ds1(x,y1)  # (B, 128, 2048) -> (B, 128, 1024)
        # x_list.append(self.conv1(y2).max(dim=-1)[0])  # (B, 128, 1024) -> (B, 512, 1024) -> (B, 512)
        # x3,y3 = self.ds2(x2,y2) # (B, 128, 1024) -> (B, 128, 512)
        # x_list.append(self.conv2(y3).max(dim=-1)[0])  # (B, 128, 512) -> (B, 512, 512) -> (B, 512)
        # z = torch.cat(x_list, dim=1)  # (B, 2048)
        # # shape_class = shape_class.view(batch_size,-1)
        # # print("shape_class:",shape_class.shape)
        # # shape_class = shape_class.view(batch_size, -1,1)
        # # shape_class = self.conv3(shape_class)  # (B, 16, 1) -> (B, 64, 1)
        # y, _ = pack([z, shape_class], 'B *')  # (B, 3136)
        # y = repeat(y, 'B C -> B C N', N=2048)  # (B, 3136) -> (B, 3136, 2048)
        # x= torch.cat([y1,y], dim=1)  # (B, C=3264, 2048)

        #新采样方法
        # x_list = []
        # y1 = self.embedding(x)  # (B, 3, 2048) -> (B, 128, 2048)
        # y1 = self.n2p_attention1(y1)  # (B, 128, 2048) -> (B, 128, 2048)
        # x_list.append(self.conv(y1).max(dim=-1)[0])  # (B, 128, 2048) -> (B, 1024, 2048) -> (B, 1024)
        # x2, y2 = self.ds1(x,y1)  # (B, 128, 2048) -> (B, 128, 1024)
        # x_list.append(self.conv1(y2).max(dim=-1)[0])  # (B, 128, 1024) -> (B, 512, 1024) -> (B, 512)
        # y2 = self.conv2(y2)  # (B, 128, 1024) -> (B, 512, 1024)
        # x3, y3 = self.ds2(x2,y2)  # (B, 512, 1024) -> (B, 512, 512)
        # x_list.append(y3.max(dim=-1)[0])  # (B, 512, 512) -> (B, 512)
        # z = torch.cat(x_list,dim=1) # (B, 2048)
        # shape_class = self.conv3(shape_class)  # (B, 16, 1) -> (B, 64, 1)
        # y, _ = pack([z, shape_class], 'B *')  # (B, 2112)
        # y = repeat(y, 'B C -> B C N', N=2048)  # (B, 2112) -> (B, 2112, 2048)
        # x= torch.cat([y1,y], dim=1)  # (B, C=2240, 2048)

        # x = self.conv4(x)  # (B, 2240, 2048) -> (B, 256, 2048)
        # x = self.dp1(x)  # (B, 256, 2048) -> (B, 256, 2048)
        # x = self.conv5(x)  # (B, 256, 2048) -> (B, 256, 2048)
        # x = self.dp2(x)  # (B, 256, 2048) -> (B, 256, 2048)
        # x = self.conv6(x)  # (B, 256, 2048) -> (B, 128, 2048)
        # x = self.conv7(x)  # (B, 128, 2048) -> (B, 50, 2048)
        x = self.backbone(x,shape_class) #(B,2240,2048)
        # x = self.head(x) #(B,50,2048)
        return x
