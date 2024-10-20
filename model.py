import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 定义LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 定义全连接层，用于输出预测值
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))

        # 取LSTM最后时间步的输出并通过全连接层
        out = self.fc(out[:, -1, :])
        return out


if __name__ == "__main__":
    input_size = 21  # 特征数
    hidden_size = 64  # 隐藏单元数
    num_layers = 3  # 隐藏层数
    output_size = 1  # 输出维度（预测的label）
    batch_size = 32  # 批次大小
    time_step = 10  # 时间步

    # 初始化模型
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)

    # 随机生成一些测试数据 (batch_size, time_step, input_size)
    test_input = torch.randn(batch_size, time_step, input_size)

    # 将模型设置为评估模式，并检查输出
    model.eval()
    with torch.no_grad():
        output = model(test_input)
        print(f"输入维度: {test_input.shape}")
        print(f"输出维度: {output.shape}")
