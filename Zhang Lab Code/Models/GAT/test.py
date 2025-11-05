import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as pyg_nn
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn


class GATBioNet(nn.Module):
    def __init__(self, n_network_nodes, edge_list, edge_MOA, in_dim, hidden_dim, out_dim, 
                 heads=4, dropout=0.1, activation='relu', device='cpu'):
        super(GATBioNet, self).__init__()

        self.device = device
        self.n_network_nodes = n_network_nodes

        # 处理 edge_list
        if isinstance(edge_list, np.ndarray):
            edge_list = edge_list.tolist()  # 确保转换
        edge_list = torch.tensor(edge_list, dtype=torch.long, device=device).T
        assert edge_list.shape[0] == 2, f"edge_index 形状错误，应为 (2, num_edges)，但得到 {edge_list.shape}"
        self.edge_index = edge_list

        # 处理 edge_MOA
        edge_MOA = torch.tensor(edge_MOA, dtype=torch.float, device=device)
        self.edge_MOA = edge_MOA

        # GAT 结构
        self.gat1 = pyg_nn.GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout, add_self_loops=False).to(device)
        self.gat2 = pyg_nn.GATConv(hidden_dim * heads, out_dim, heads=1, dropout=dropout, add_self_loops=False).to(device)

        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, X):
        X = X.to(self.device)
        X = self.activation(self.gat1(X, self.edge_index))
        X = self.gat2(X, self.edge_index)
        return X



# ========================
# 初始化数据
# ========================

num_nodes = 7668

# 生成随机的边索引（2, num_edges），以适应 PyG 格式
num_edges = 20000  # 你可以根据数据集调整
edge_list = torch.randint(0, num_nodes, (2, num_edges)).tolist()

# 生成随机的 edge_MOA（模拟边的特性），与边数量匹配
edge_MOA = torch.randint(-1, 2, (num_edges,)).tolist()

# 初始化模型
model = GATBioNet(
    n_network_nodes=num_nodes, 
    edge_list=edge_list,  # 传入边索引
    edge_MOA=edge_MOA,    # 传入边 MOA 信息
    in_dim=num_nodes, 
    hidden_dim=256, 
    out_dim=num_nodes, 
    heads=4, 
    dropout=0.1, 
    activation='relu',
    device=device
).to(device)


# ========================
# 测试不同 batch_size
# ========================

X_input1 = torch.randn(20, 7668).to(device)  # batch_size=20
X_input2 = torch.randn(4892, 7668).to(device)  # batch_size=4892
X_input3 = torch.randn(1024, 7668).to(device)  # batch_size=1024

output1 = model(X_input1)
output2 = model(X_input2)
output3 = model(X_input3)

print("输出形状 (batch_size=20):", output1.shape)  # 预期: (20, 7668)
print("输出形状 (batch_size=4892):", output2.shape)  # 预期: (4892, 7668)
print("输出形状 (batch_size=1024):", output3.shape)  # 预期: (1024, 7668)

