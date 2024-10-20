import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataset, get_stock
from torch.utils.data import DataLoader, TensorDataset

from dataset import create_sequences
from model import LSTMModel

def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()

    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


stock_code = "sh.600048"
index_code = "sh.000001"

# 1. data
df, mean_and_std, date_and_code = get_stock(stock_code, index_code)

feature, label = get_dataset(df)

# 超参数
input_size = len(feature[1])  # features numbers
hidden_size = 64
num_layers = 3
output_size = 1
time_step = 10

batch_size = 32
learning_rate = 0.01
num_epochs = 100
device = try_gpu()

# 数据准备，构造时间步为10的样本
X, Y = create_sequences(feature, label, time_step)

# 转换为Tensor并创建DataLoader
dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 初始化模型
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    for fea, lab in dataloader:
        fea, lab = fea.to(device), lab.to(device)

        # 前向传播
        outputs = model(fea)
        loss = criterion(outputs, lab)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 保存模型
torch.save(model.state_dict(), './model_param/lstm_model.pth')
