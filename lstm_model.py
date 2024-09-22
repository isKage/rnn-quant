import torch
import torch.nn as nn


class myLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(myLSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer 在LSTM后再加一个全连接层，因为是回归问题，所以不能在线性层后加激活函数
        self.outputs = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        # 这里x.size(0)就是batch_size

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # One time step
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.outputs(out)

        return out

if __name__ == '__main__':
    model = myLSTM(13, 32, 2, 1)
    # print(model)
    x = torch.randn(1, 1, 13)
    y = model(x)
    print(y)
    print(y.shape)



'''
nn.LSTM(
    input_size, 
    hidden_size, 
    num_layers=1, 
    bias=True, 
    batch_first=False, 
    dropout=0.0, 
    bidirectional=False, 
    proj_size=0, 
    device=None, 
    dtype=None
)

input_size – 输入的特征数量

hidden_size – 隐藏状态的特征数量（H的元素个数）

num_layers – 隐藏状态层层数，即深度 Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. Default: 1

bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True

batch_first – If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature). Note that this does not apply to hidden or cell states. See the Inputs/Outputs sections below for details. Default: False

dropout – If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout. Default: 0

bidirectional – 双头与否 If True, becomes a bidirectional LSTM. Default: False

proj_size – 是否梯度裁剪 If > 0, will use LSTM with projections of corresponding size. Default: 0
'''
