import torch, math
from core.models.utils import ops
from torch import nn
from einops import rearrange, repeat
import torch.nn.init as init
import torch.nn.functional as F

class Transform_Net(nn.Module):
    def __init__(self):
        super(Transform_Net, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Sequential(nn.Linear(1024, 512, bias=False),nn.BatchNorm1d(512),nn.LeakyReLU(negative_slope=0.2))
        self.linear2 = nn.Sequential(nn.Linear(512, 256, bias=False),nn.BatchNorm1d(256),nn.LeakyReLU(negative_slope=0.2))
        self.transform = nn.Linear(256, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)
        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)
        x = self.linear1(x)    # (batch_size, 1024) -> (batch_size, 512)
        x = self.linear2(x)    # (batch_size, 512) -> (batch_size, 256)
        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)
        return x

class stage1(nn.Module):
    def __init__(self):
        super(stage1, self).__init__()
        self.K = 16
        self.group_type = 'center_diff'
        self.conv1 = nn.Sequential(nn.Conv2d(6, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        # self.conv1 = nn.Sequential(nn.Conv2d(20, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        # self.conv2 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        # self.conv3 = nn.Sequential(nn.Conv2d(128, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        # self.conv4 = nn.Sequential(nn.Conv2d(256, 256, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))

    def forward(self, x):
        x_list = []
        # neighbor = ops.group(x, self.K, 'neighbor')  # (B,C=3,N,K)
        # y = ops.local_features(neighbor)   #（B,C=7,N)
        # x = torch.cat([x,y],dim=1)      # (B,C=10,N)
        x = ops.group(x, self.K, self.group_type)  # (B, C=3, N) -> (B, C=6, N, K)
        x = self.conv1(x)  # (B, C=6, N, K) -> (B, C=128, N, K)
        x = self.conv2(x)  # (B, C=128, N, K) -> (B, C=64, N, K)
        x = x.max(dim=-1, keepdim=False)[0]  # (B, C=64, N, K) -> (B, C=64, N)
        x_list.append(x)
        x = ops.group(x, self.K, self.group_type)  # (B, C=64, N) -> (B, C=128, N, K)
        x = self.conv3(x)  # (B, C=128, N, K) -> (B, C=128, N, K)
        x = self.conv4(x)  # (B, C=128, N, K) -> (B, C=64, N, K)
        x = x.max(dim=-1, keepdim=False)[0]  # (B, C=64, N, K) -> (B, C=64, N)
        x_list.append(x)
        x = torch.cat(x_list, dim=1)  # (B, C=128, N)
        return x

class down(nn.Module):
    def __init__(self,npts_ds):
        super(down, self).__init__()
        self.npts_ds = npts_ds  # number of downsampled points
        self.K = 16
        self.group_type = 'center_diff'
        # self.conv1 = nn.Sequential(nn.Conv2d(6, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        # self.conv2 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        # self.conv3 = nn.Sequential(nn.Conv2d(128, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        # self.conv4 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        # self.conv1 = nn.Sequential(nn.Conv2d(20, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        # self.conv2 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        # self.conv3 = nn.Sequential(nn.Conv2d(128, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        # self.conv4 = nn.Sequential(nn.Conv2d(256, 256, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))

    def forward(self, x,y):
        # x_list = []
        #x(B,3,N) y(B,C,N)
        neighbor = ops.group(x, self.K, 'neighbor')  # (B,C=3,N,K)
        # top_k_indices = ops.down_index(neighbor,self.npts_ds)  #(B,M)
        self.idx = ops.down_index(neighbor,self.npts_ds) #(B,M)
        top_k_xyz = torch.gather(x, 2, self.idx.unsqueeze(1).repeat(1, x.shape[1], 1)) #(B,3,M)
        top_k_points = torch.gather(y, 2, self.idx.unsqueeze(1).repeat(1, y.shape[1], 1)) #(B,C,M)
        return top_k_xyz,top_k_points

class down_feature(nn.Module):
    def __init__(self,npts_ds):
        super(down_feature, self).__init__()
        self.npts_ds = npts_ds  # number of downsampled points

    def forward(self, x,y):

        # 标准版本
        # x(B,3,1024) y(B,512,1024)
        # self.idx = torch.argmax(y,dim=2)#(B,512)
        # top_k_xyz = torch.gather(x, 2, self.idx.unsqueeze(1).repeat(1, x.shape[1], 1))  # (B,3,512)
        # top_k_points = torch.gather(y, 2, self.idx.unsqueeze(1).repeat(1, y.shape[1], 1))  # (B,512,512)

        # 修正版本
        z = F.softmax(y, dim=2)
        z = y.max(dim=1)[0]  # (B,512,1024) -> (B,1024)
        # z = torch.sum(z, dim=1) # (B,512,1024) -> (B,1024)
        _, index = torch.topk(z, self.npts_ds, dim=1, largest=True, sorted=True)  # (B,512)
        self.idx = index
        top_k_xyz = torch.gather(x, 2, self.idx.unsqueeze(1).repeat(1, x.shape[1], 1))  # (B,3,512)
        top_k_points = torch.gather(y, 2, self.idx.unsqueeze(1).repeat(1, y.shape[1], 1))  # (B,512,512)

        return top_k_xyz,top_k_points

class Attention(nn.Module):
    def __init__(self, channels):
        super(Attention, self).__init__()
        self.heads = 4
        self.K = 8
        self.group_type = 'diff'
        self.q_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.k_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.v_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.ff = nn.Sequential(nn.Conv1d(channels, 256, 1, bias=False), nn.LeakyReLU(0.2),
                                nn.Conv1d(256, channels, 1, bias=False))
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        neighbors = ops.group(x, self.K, self.group_type)  # (B, C, N) -> (B, C, N, K)
        q = self.q_conv(rearrange(x, 'B C N -> B C N 1')).contiguous()  # (B, C, N) -> (B, C, N, 1)
        q = self.split_heads(q, self.heads)  # (B, C, N, 1) -> (B, H, N, 1, D)
        k = self.k_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        k = self.split_heads(k, self.heads)  # (B, C, N, K) -> (B, H, N, K, D)
        v = self.v_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        v = self.split_heads(v, self.heads)  # (B, C, N, K) -> (B, H, N, K, D)
        energy = q @ rearrange(k,
                               'B H N K D -> B H N D K').contiguous()  # (B, H, N, 1, D) @ (B, H, N, D, K) -> (B, H, N, 1, K)
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)  # (B, H, N, 1, K) -> (B, H, N, 1, K)
        tmp = rearrange(attention @ v,
                        'B H N 1 D -> B (H D) N').contiguous()  # (B, H, N, 1, K) @ (B, H, N, K, D) -> (B, H, N, 1, D) -> (B, C=H*D, N)
        x = self.bn1(x + tmp)  # (B, C, N) + (B, C, N) -> (B, C, N)
        tmp = self.ff(x)  # (B, C, N) -> (B, C, N)
        x = self.bn2(x + tmp)  # (B, C, N) + (B, C, N) -> (B, C, N)
        return x

    @staticmethod
    def split_heads(x, heads):
        x = rearrange(x, 'B (H D) N K -> B H N K D', H=heads).contiguous()  # (B, C, N, K) -> (B, H, N, K, D)
        return x

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c
        x_k = self.k_conv(x)# b, c, n
        x_v = self.v_conv(x)
        energy = x_q @ x_k # b, n, n
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = x_v @ attention # b, c, n
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.K = 32
        self.group_type = 'center_diff'
        self.transform_net = Transform_Net()
        self.sa1 = SA_Layer(64)
        self.sa2 = SA_Layer(64)

        #原始特征提取
        # self.conv1 = nn.Sequential(nn.Conv2d(6, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        # self.conv2 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        # self.conv3 = nn.Sequential(nn.Conv2d(128, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        # self.conv4 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))

        #多尺度动态特征提取
        self.conv1 = nn.Sequential(nn.Conv2d(6, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(128, 32, 1, bias=False), nn.BatchNorm2d(32), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 32, 1, bias=False), nn.BatchNorm2d(32), nn.LeakyReLU(0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(6, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv6 = nn.Sequential(nn.Conv2d(128, 32, 1, bias=False), nn.BatchNorm2d(32), nn.LeakyReLU(0.2))
        self.conv7 = nn.Sequential(nn.Conv2d(64, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv8 = nn.Sequential(nn.Conv2d(128, 32, 1, bias=False), nn.BatchNorm2d(32), nn.LeakyReLU(0.2))


    def forward(self, x):

        # x0 = ops.group(x, self.K, self.group_type)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        # t = self.transform_net(x0)  # (batch_size, 3, 3)
        # x = x.transpose(2, 1)  # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        # x = torch.bmm(x, t)  # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        # x = x.transpose(2, 1)  # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        #原始特征提取
        # x_list = []
        # x = ops.group(x, self.K, self.group_type)  # (B, C=3, N) -> (B, C=6, N, K)
        # x = self.conv1(x)  # (B, C=6, N, K) -> (B, C=128, N, K)
        # x = self.conv2(x)  # (B, C=128, N, K) -> (B, C=64, N, K)
        # x = x.max(dim=-1, keepdim=False)[0]  # (B, C=64, N, K) -> (B, C=64, N)
        # x = self.sa1(x)
        # x_list.append(x)
        # x = ops.group(x, self.K, self.group_type)  # (B, C=64, N) -> (B, C=128, N, K)
        # x = self.conv3(x)  # (B, C=128, N, K) -> (B, C=128, N, K)
        # x = self.conv4(x)  # (B, C=128, N, K) -> (B, C=64, N, K)
        # x = x.max(dim=-1, keepdim=False)[0]  # (B, C=64, N, K) -> (B, C=64, N)
        # x = self.sa2(x)
        # x_list.append(x)
        # x = torch.cat(x_list, dim=1)  # (B, C=128, N)

        #多尺度动态特征提取
        x_list = []
        x1 = ops.group(x, 15, self.group_type)  # (B, C=3, N) -> (B, C=6, N, K)
        x1 = self.conv1(x1)  # (B, C=6, N, K) ->   (B, C=128, N, K)
        x1 = self.conv2(x1)  # (B, C=128, N, K) -> (B, C=32, N, K)
        x1 = x1.max(dim=-1, keepdim=False)[0]   #(B, C=32, N, K) -> (B, C=32, N)
        x_list.append(x1)
        y1 = ops.group(x1, 15, self.group_type)  # (B, C=32, N) -> (B, C=64, N, K)
        y1 = self.conv3(y1)  # (B, C=64, N, K) -> (B, C=128, N, K)
        y1 = self.conv4(y1)  # (B, C=128, N, K) -> (B, C=32, N, K)
        y1 = y1.max(dim=-1, keepdim=False)[0]   # (B, C=32, N, K) -> (B, C=32, N)
        x_list.append(y1)
        x2 = ops.group(x, 20, self.group_type)  # (B, C=3, N) -> (B, C=6, N, K)
        x2 = self.conv5(x2)  # (B, C=6, N, K) -> (B, C=128, N, K)
        x2 = self.conv6(x2)  # (B, C=128, N, K) -> (B, C=32, N, K)
        x2 = x2.max(dim=-1, keepdim=False)[0]  # (B, C=32, N, K) -> (B, C=32, N)
        x_list.append(x2)
        y2 = ops.group(x2, 20, self.group_type)  # (B, C=32, N) -> (B, C=64, N, K)
        y2 = self.conv7(y2)  # (B, C=64, N, K) -> (B, C=128, N, K)
        y2 = self.conv8(y2)  # (B, C=128, N, K) -> (B, C=32, N, K)
        y2 = y2.max(dim=-1, keepdim=False)[0]  # (B, C=32, N, K) -> (B, C=32, N)
        x_list.append(y2)
        x = torch.cat(x_list, dim=1)  # (B, C=128, N)


        # x_list = []
        # neighbor = ops.group(x, self.K, 'neighbor')  # (B,C=3,N,K)
        # y = ops.local_features(neighbor)   #（B,C=7,N)
        # x = torch.cat([x,y],dim=1)      #(B,C=10,N)
        # x = ops.group(x, self.K, self.group_type)  # (B, C=10, N) -> (B, C=20, N, K)
        # x = self.conv1(x)  # (B, C=20, N, K) -> (B, C=64, N, K)
        # x = x.max(dim=-1, keepdim=False)[0]  # (B, C=64, N, K) -> (B, C=64, N)
        # x_list.append(x)
        # x = ops.group(x, self.K, self.group_type)  # (B, C=64, N) -> (B, C=128, N, K)
        # x = self.conv2(x)  # (B, C=128, N, K) -> (B, C=64, N, K)
        # x = x.max(dim=-1, keepdim=False)[0]  # (B, C=64, N, K) -> (B, C=64, N)
        # x_list.append(x)
        # x = ops.group(x, self.K, self.group_type)  # (B, C=64, N) -> (B, C=128, N, K)
        # x = self.conv3(x)  # (B, C=128, N, K) -> (B, C=128, N, K)
        # x = x.max(dim=-1, keepdim=False)[0]  # (B, C=128, N, K) -> (B, C=128, N)
        # x_list.append(x)
        # x = ops.group(x, self.K, self.group_type)  # (B, C=128, N) -> (B, C=256, N, K)
        # x = self.conv4(x)  # (B, C=256, N, K) -> (B, C=256, N, K)
        # x = x.max(dim=-1, keepdim=False)[0]  # (B, C=256, N, K) -> (B, C=256, N)
        # x_list.append(x)
        # x = torch.cat(x_list, dim=1)  # (B, C=512, N)

        return x


class N2PAttention(nn.Module):
    def __init__(self):
        super(N2PAttention, self).__init__()
        self.heads = 4
        self.K = 8
        self.group_type = 'diff'
        self.q_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.k_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.v_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.ff = nn.Sequential(nn.Conv1d(128, 512, 1, bias=False), nn.LeakyReLU(0.2), nn.Conv1d(512, 128, 1, bias=False))
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x):
        neighbors = ops.group(x, self.K, self.group_type)  # (B, C, N) -> (B, C, N, K)
        q = self.q_conv(rearrange(x, 'B C N -> B C N 1')).contiguous()  # (B, C, N) -> (B, C, N, 1)
        q = self.split_heads(q, self.heads)  # (B, C, N, 1) -> (B, H, N, 1, D)
        k = self.k_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        k = self.split_heads(k, self.heads)  # (B, C, N, K) -> (B, H, N, K, D)
        v = self.v_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        v = self.split_heads(v, self.heads)  # (B, C, N, K) -> (B, H, N, K, D)
        energy = q @ rearrange(k, 'B H N K D -> B H N D K').contiguous()  # (B, H, N, 1, D) @ (B, H, N, D, K) -> (B, H, N, 1, K)
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)  # (B, H, N, 1, K) -> (B, H, N, 1, K)
        tmp = rearrange(attention@v, 'B H N 1 D -> B (H D) N').contiguous()  # (B, H, N, 1, K) @ (B, H, N, K, D) -> (B, H, N, 1, D) -> (B, C=H*D, N)
        x = self.bn1(x + tmp)  # (B, C, N) + (B, C, N) -> (B, C, N)
        tmp = self.ff(x)  # (B, C, N) -> (B, C, N)
        x = self.bn2(x + tmp)  # (B, C, N) + (B, C, N) -> (B, C, N)
        return x

    @staticmethod
    def split_heads(x, heads):
        x = rearrange(x, 'B (H D) N K -> B H N K D', H=heads).contiguous()  # (B, C, N, K) -> (B, H, N, K, D)
        return x


# class GlobalDownSample(nn.Module):
#     def __init__(self, npts_ds):
#         super(GlobalDownSample, self).__init__()
#         self.npts_ds = npts_ds
#         self.q_conv = nn.Conv1d(128, 128, 1, bias=False)
#         self.k_conv = nn.Conv1d(128, 128, 1, bias=False)
#         self.v_conv = nn.Conv1d(128, 128, 1, bias=False)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x):
#         q = self.q_conv(x)  # (B, C, N) -> (B, C, N)
#         k = self.k_conv(x)  # (B, C, N) -> (B, C, N)
#         v = self.v_conv(x)  # (B, C, N) -> (B, C, N)
#         energy = rearrange(q, 'B C N -> B N C').contiguous() @ k  # (B, N, C) @ (B, C, N) -> (B, N, N)
#         scale_factor = math.sqrt(q.shape[-2])
#         attention = self.softmax(energy / scale_factor)  # (B, N, N) -> (B, N, N)
#         selection = torch.sum(attention, dim=-2)  # (B, N, N) -> (B, N)
#         self.idx = selection.topk(self.npts_ds, dim=-1)[1]  # (B, N) -> (B, M)
#         scores = torch.gather(attention, dim=1, index=repeat(self.idx, 'B M -> B M N', N=attention.shape[-1]))  # (B, N, N) -> (B, M, N)
#         v = scores @ rearrange(v, 'B C N -> B N C').contiguous()  # (B, M, N) @ (B, N, C) -> (B, M, C)
#         out = rearrange(v, 'B M C -> B C M').contiguous()  # (B, M, C) -> (B, C, M)
#         return out


# class LocalDownSample(nn.Module):
#     def __init__(self, npts_ds):
#         super(LocalDownSample, self).__init__()
#         self.npts_ds = npts_ds  # number of downsampled points
#         self.K = 8  # number of neighbors
#         self.group_type = 'diff'
#         self.q_conv = nn.Conv2d(128, 128, 1, bias=False)
#         self.k_conv = nn.Conv2d(128, 128, 1, bias=False)
#         self.v_conv = nn.Conv2d(128, 128, 1, bias=False)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x):
#         neighbors = ops.group(x, self.K, self.group_type)  # (B, C, N) -> (B, C, N, K)
#         q = self.q_conv(rearrange(x, 'B C N -> B C N 1')).contiguous()  # (B, C, N) -> (B, C, N, 1)
#         q = rearrange(q, 'B C N 1 -> B N 1 C').contiguous()  # (B, C, N, 1) -> (B, N, 1, C)
#         k = self.k_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
#         k = rearrange(k, 'B C N K -> B N C K').contiguous()  # (B, C, N, K) -> (B, N, C, K)
#         v = self.v_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
#         v = rearrange(v, 'B C N K -> B N K C').contiguous()  # (B, C, N, K) -> (B, N, K, C)
#         energy = q @ k  # (B, N, 1, C) @ (B, N, C, K) -> (B, N, 1, K)
#         scale_factor = math.sqrt(q.shape[-1])
#         attention = self.softmax(energy / scale_factor)  # (B, N, 1, K) -> (B, N, 1, K)
#         selection = rearrange(torch.std(attention, dim=-1, unbiased=False), 'B N 1 -> B N').contiguous()  # (B, N, 1, K) -> (B, N, 1) -> (B, N)
#         self.idx = selection.topk(self.npts_ds, dim=-1)[1]  # (B, N) -> (B, M)
#         scores = torch.gather(attention, dim=1, index=repeat(self.idx, 'B M -> B M 1 K', K=attention.shape[-1]))  # (B, N, 1, K) -> (B, M, 1, K)
#         v = torch.gather(v, dim=1, index=repeat(self.idx, 'B M -> B M K C', K=v.shape[-2], C=v.shape[-1]))  # (B, N, K, C) -> (B, M, K, C)
#         out = rearrange(scores@v, 'B M 1 C -> B C M').contiguous()  # (B, M, 1, K) @ (B, M, K, C) -> (B, M, 1, C) -> (B, C, M)
#         return out


# class UpSample(nn.Module):
#     def __init__(self):
#         super(UpSample, self).__init__()
#         self.q_conv = nn.Conv1d(128, 128, 1, bias=False)
#         self.k_conv = nn.Conv1d(128, 128, 1, bias=False)
#         self.v_conv = nn.Conv1d(128, 128, 1, bias=False)
#         self.skip_link = nn.Conv1d(128, 128, 1, bias=False)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, pcd_up, pcd_down):
#         q = self.q_conv(pcd_up)  # (B, C, N) -> (B, C, N)
#         k = self.k_conv(pcd_down)  # (B, C, M) -> (B, C, M)
#         v = self.v_conv(pcd_down)  # (B, C, M) -> (B, C, M)
#         energy = rearrange(q, 'B C N -> B N C').contiguous() @ k  # (B, N, C) @ (B, C, M) -> (B, N, M)
#         scale_factor = math.sqrt(q.shape[-2])
#         attention = self.softmax(energy / scale_factor)  # (B, N, M) -> (B, N, M)
#         x = attention @ rearrange(v, 'B C M -> B M C').contiguous()  # (B, N, M) @ (B, M, C) -> (B, N, C)
#         x = rearrange(x, 'B N C -> B C N').contiguous()  # (B, N, C) -> (B, C, N)
#         x = self.skip_link(pcd_up) + x  # (B, C, N) + (B, C, N) -> (B, C, N)
#         return x
