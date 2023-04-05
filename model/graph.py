import torch


#Used in stage 1 (ANFL)
def normalize_digraph(A): # 作用是将A矩阵进行归一化，即将A矩阵中的每个元素除以对应节点的度，得到归一化的A矩阵，这里的A矩阵是一个batch的，所以需要对每个batch中的A矩阵进行归一化 
    b, n, _ = A.shape
    # print(f'[model/graph.py] 正在执行 normalize_digraph 函数 b = {b}, n = {n}, _ = {_}') # b = 3, n = 12, _ = 12
    node_degrees = A.detach().sum(dim = -1) # 求出每个节点的度，即每个节点的出度和入度之和
    degs_inv_sqrt = node_degrees ** -0.5 # 求出每个节点的度的倒数的平方根
    # print(f'[model/graph.py] 正在执行 normalize_digraph degs_inv_sqrt = {degs_inv_sqrt}')
    """ degs_inv_sqrt = tensor([[0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                                [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                                [0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000]]) """
    norm_degs_matrix = torch.eye(n) # 创建一个单位矩阵 torch.eye 用于创建单位矩阵 例如 torch.eye(2) = [[1,0],[0,1]]，torch.eye(3) = [[1,0,0],[0,1,0],[0,0,1]] 单位矩阵指的是对角线上的元素都为1，其余元素都为0的矩阵  
    dev = A.get_device() # 获取A矩阵所在的设备，即GPU或者CPU
    if dev >= 0:
        norm_degs_matrix = norm_degs_matrix.to(dev) 
    norm_degs_matrix = norm_degs_matrix.view(1, n, n) * degs_inv_sqrt.view(b, n, 1) # 将单位矩阵扩展为batch的单位矩阵，然后将单位矩阵与每个节点的度的倒数的平方根相乘，得到归一化的单位矩阵
    norm_A = torch.bmm(torch.bmm(norm_degs_matrix,A),norm_degs_matrix) # 将归一化的单位矩阵与A矩阵相乘，得到归一化的A矩阵，torch.bmm 是指 batch matrix multiplication，即对两个batch的矩阵进行矩阵乘法 
    return norm_A


#Used in stage 2 (MEFL)
def create_e_matrix(n):
    end = torch.zeros((n*n,n))
    for i in range(n):
        end[i * n:(i + 1) * n, i] = 1
    start = torch.zeros(n, n)
    for i in range(n):
        start[i, i] = 1
    start = start.repeat(n,1)
    return start,end


