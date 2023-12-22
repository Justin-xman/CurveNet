import torch
import torch.nn.functional as F

def down_index(x,k):
    # 假设输入数据为 x，维度为 (B, C, N, K)
    x = x.permute(0,2,3,1)
    #数据维度变为（B,N,K,C)
    B, N, K, C = x.size()
    # 把输入数据 reshape 为 (B * N, K, C)，以便后面计算协方差矩阵
    x_reshaped = x.reshape(B * N, K, C)

    # 计算每个点的近邻点的协方差矩阵
    x_centered = x_reshaped - x_reshaped.mean(dim=1, keepdim=True)
    cov = torch.bmm(x_centered.transpose(1, 2), x_centered) / (K - 1)

    # 计算每个协方差矩阵的特征值
    eigvals1, _ = torch.linalg.eigh(cov.to(torch.float32))
    epsilon_to_add = 1e-8
    eigvals = torch.max(eigvals1, torch.tensor(epsilon_to_add))
    # eigvals =eigvals1 / torch.sum(eigvals1,dim=1,keepdim=True)
    x1 = eigvals[:,0]
    x2 = eigvals[:,1]
    x3 = eigvals[:,2]
    curvature = x3 / (x1 + x2 + x3)
    curvature = curvature.reshape(B,N)
    _, rank = torch.sort(curvature, dim=1, descending=True)
    top_k_indices = rank[:, :k]
    return top_k_indices


# def avg_pool_neighbor(x,k):
#     a,b,c = x.shape
#     x =x.reshape(a*c,b)
#     #print(x.shape)
#     edge_index = knn_graph(x,k,loop=True)
#     data = Data(x=x,edge_index=edge_index)
#     x,edge_index= data.x,data.edge_index
#     #edge_index, _ = add_self_loops(edge_index, num_nodes=data.num_nodes)
#     row , col = edge_index
#     x = scatter(x[row],col,dim=0,dim_size=data.num_nodes,reduce='mean')
#     #print(x.shape)
#     x = x.reshape(a,b,c)
#     return x

def local_features(x):
    # 假设输入数据为 x，维度为 (B, C, N, K)
    x = x.permute(0,2,3,1)
    #数据维度变为（B,N,K,C)
    B, N, K, C = x.size()

    # 把输入数据 reshape 为 (B * N, K, C)，以便后面计算协方差矩阵
    x_reshaped = x.reshape(B * N, K, C)
    # 计算每个点的近邻点的协方差矩阵
    x_centered = x_reshaped - x_reshaped.mean(dim=1, keepdim=True)
    cov = torch.bmm(x_centered.transpose(1, 2), x_centered) / (K - 1)

    # 计算每个协方差矩阵的特征值
    eigvals1, _ = torch.linalg.eigh(cov.to(torch.float32))
    # eigvals = F.normalize(eigvals1, p=2, dim=1)
    epsilon_to_add = 1e-6
    eigvals1 = torch.max(eigvals1, torch.tensor(epsilon_to_add))
    eigvals = eigvals1 / torch.sum(eigvals1, dim=1, keepdim=True)
    #print(eigvals[:100,:])
    x1 = eigvals[:,0]
    x2 = eigvals[:,1]
    x3 = eigvals[:,2]

    linearity = (x1 - x2) / x1

    # max_linearity = torch.max(linearity)
    # min_linearity = torch.min(linearity)
    # linearity = (linearity - min_linearity) / (max_linearity - min_linearity)
    # print(linearity[:100])

    planarity = (x2 - x3) / x1

    # max_planarity = torch.max(planarity)
    # min_planarity = torch.min(planarity)
    # planarity = (planarity - min_planarity) / (max_planarity - min_planarity)
    # print(planarity[:100])

    scattering = x3 / x1

    # max_scattering = torch.max(scattering)
    # min_scattering = torch.min(scattering)
    # scattering = (scattering - min_scattering) / (max_scattering - min_scattering)
    # print(scattering[:100])
    omnivariance = (x1 * x2 * x3) ** (1 / 3)
    anisotropy = (x1 - x3) / x1
    eigenentropy = -(x1 * torch.log(x1) + x2 * torch.log(x2) + x3 * torch.log(x3))
    # sum = x1+x2+x3
    curvature = x3 / (x1 + x2 + x3)
    all = torch.column_stack((linearity,planarity,scattering,omnivariance,anisotropy,eigenentropy,curvature))
    all_normalized = F.normalize(all, p=2, dim=1)
    # print(all_normalized[:100,:])
    # 把特征值 reshape 回 (B, N, C) 的形状
    all = all_normalized.reshape(B, N, -1)
    all = all.permute(0,2,1) #(B,C,N)

    return all



def index_points(points, idx):
    """
    :param points: points.shape == (B, N, C)
    :param idx: idx.shape == (B, N, K)
    :return:indexed_points.shape == (B, N, K, C)
    """
    raw_shape = idx.shape
    idx = idx.reshape(raw_shape[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.shape[-1]))
    return res.view(*raw_shape, -1)


def knn(a, b, k):
    """
    :param a: a.shape == (B, N, C)
    :param b: b.shape == (B, M, C)
    :param k: int
    """
    inner = -2 * torch.matmul(a, b.transpose(2, 1))  # inner.shape == (B, N, M)
    aa = torch.sum(a**2, dim=2, keepdim=True)  # aa.shape == (B, N, 1)
    bb = torch.sum(b**2, dim=2, keepdim=True)  # bb.shape == (B, M, 1)
    pairwise_distance = -aa - inner - bb.transpose(2, 1)  # pairwise_distance.shape == (B, N, M)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # idx.shape == (B, N, K)
    return idx


def select_neighbors(pcd, K, neighbor_type):
    pcd = pcd.permute(0, 2, 1)  # pcd.shape == (B, N, C)
    idx = knn(pcd, pcd, K)  # idx.shape == (B, N, K)
    neighbors = index_points(pcd, idx)  # neighbors.shape == (B, N, K, C)
    if neighbor_type == 'neighbor':
        neighbors = neighbors.permute(0, 3, 1, 2)  # output.shape == (B, C, N, K)
    elif neighbor_type == 'diff':
        diff = neighbors - pcd[:, :, None, :]  # diff.shape == (B, N, K, C)
        neighbors = diff.permute(0, 3, 1, 2)  # output.shape == (B, C, N, K)
    else:
        raise ValueError(f'neighbor_type should be "neighbor" or "diff", but got {neighbor_type}')
    return neighbors


def group(pcd, K, group_type):
    if group_type == 'neighbor':
        neighbors = select_neighbors(pcd, K, 'neighbor')  # neighbors.shape == (B, C, N, K)
        output = neighbors  # output.shape == (B, C, N, K)
    elif group_type == 'diff':
        diff = select_neighbors(pcd, K, 'diff')  # diff.shape == (B, C, N, K)
        output = diff  # output.shape == (B, C, N, K)
    elif group_type == 'center_neighbor':
        neighbors = select_neighbors(pcd, K, 'neighbor')   # neighbors.shape == (B, C, N, K)
        output = torch.cat([pcd[:, :, :, None].repeat(1, 1, 1, K), neighbors], dim=1)  # output.shape == (B, 2C, N, K)
    elif group_type == 'center_diff':
        diff = select_neighbors(pcd, K, 'diff')  # diff.shape == (B, C, N, K)
        output = torch.cat([pcd[:, :, :, None].repeat(1, 1, 1, K), diff], dim=1)  # output.shape == (B, 2C, N, K)
    else:
        raise ValueError(f'group_type should be neighbor, diff, center_neighbor or center_diff, but got {group_type}')
    return output.contiguous()
