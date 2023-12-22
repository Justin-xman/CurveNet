from ..utils.layers import Embedding,down,down_feature,SA_Layer
from torch import nn
from einops import pack



class backbone(nn.Module):
    def __init__(self):
        super(backbone, self).__init__()
        self.embedding = Embedding()
        # self.ds1 = down(1024)  # 2048 pts -> 1024 pts
        # # self.ds2 = down(512)  # 1024 pts -> 512 pts
        # self.ds2 = down_feature(512)
        self.ds1 = down_feature(1024)
        self.ds2 = down(512)
        self.sa1 = SA_Layer(128)
        self.sa2 = SA_Layer(128)
        self.conv = nn.Sequential(nn.Conv1d(128, 1024, 1, bias=False), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2))
        self.conv1 = nn.Sequential(nn.Conv1d(128, 512, 1, bias=False), nn.BatchNorm1d(512), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(128, 512, 1, bias=False), nn.BatchNorm1d(512), nn.LeakyReLU(0.2))


    def forward(self, x):
        #MSADGC+SPFF
        self.res_link_list = []
        y = self.embedding(x)  # (B, 3, 2048) -> (B, 128, 2048)
        y = self.conv(y).max(dim=-1)[0]
        # self.res_link_list.append(self.conv(y).max(dim=-1)[0])  # (B, 128, 2048) -> (B, 1024, 2048) -> (B, 1024)
        # y = self.sa1(y)  # (B, 128, 2048) -> (B, 128, 2048)
        # x, y = self.ds1(x, y)  # (B, 128, 2048) -> (B, 128, 1024)
        # self.res_link_list.append(self.conv1(y).max(dim=-1)[0])  # (B, 128, 1024) -> (B, 1024, 1024) -> (B, 1024)
        # # y = self.conv2(y)  # (B, 128, 1024) -> (B, 512, 1024)
        # # x, y = self.ds2(x, y)  # (B, 512, 1024) -> (B, 512, 512)
        # # self.res_link_list.append(y.max(dim=-1)[0])  # (B, 512, 512) -> (B, 512)
        # y = self.sa2(y)   # (B, 128, 2048) -> (B, 128, 2048)
        # x, y = self.ds2(x, y)  # (B, 128, 1024) -> (B, 128, 512)
        # self.res_link_list.append(self.conv2(y).max(dim=-1)[0])  # (B, 128, 512) -> (B, 512, 512) -> (B, 512)
        # x, ps = pack(self.res_link_list, 'B *')  # (B, 2048)

        #只有特征MSADGC
        # self.res_link_list = []
        # y = self.embedding(x)  # (B, 3, 2048) -> (B, 128, 2048)
        # y = self.conv(y)        # (B, 128, 2048) -> (B, 1024, 2048)
        # self.res_link_list.append(y.max(dim=-1)[0])             # (B, 1024, 2048) -> (B, 1024)
        # self.res_link_list.append(y.mean(dim=-1))   # (B, 1024, 2048) -> (B, 1024)
        # x, ps = pack(self.res_link_list, 'B *')  # (B, 2048)
        return y

class head(nn.Module):
    def __init__(self):
        super(head, self).__init__()
        self.linear1 = nn.Sequential(nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2))
        self.linear2 = nn.Sequential(nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2))
        self.linear3 = nn.Linear(256, 40)
        self.dp1 = nn.Dropout(0.5)
        self.dp2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.linear1(x)  # (B, 3072) -> (B, 1024)
        x = self.dp1(x)  # (B, 1024) -> (B, 1024)
        x = self.linear2(x)  # (B, 1024) -> (B, 256)
        x = self.dp2(x)  # (B, 256) -> (B, 256)
        x = self.linear3(x)  # (B, 256) -> (B, 40)
        return x

class CurveNet(nn.Module):
    def __init__(self):
        super(CurveNet, self).__init__()
        self.backbone = backbone()
        self.head = head()

    def forward(self, x):

        x = self.backbone(x) #(B,2560)
        # x = self.head(x) #(B,40)
        return x
