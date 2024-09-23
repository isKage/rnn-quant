import numpy as np

from lstm_model import *
from data_pre import train_and_test, data_loader


def train_lstm_for_stock(
        code: str,
        time: int,
        dir: str,
        time_step=10,
        train_size=0.8,
        hidden_dim=32,
        num_layers=2,
        learning_rate=0.01,
        num_epochs=200,
):
    """
    训练
    :param code: 股票代码
    :param time: 延续时间
    :param dir: 模型存储路径
    :param time_step: 时间步
    :param train_size: 训练集划分
    :param hidden_dim: 隐藏层的神经元个数
    :param num_layers: LSTM的层数
    :param learning_rate: 学习率
    :param num_epochs: 设定数据遍历次数
    :return:
    """
    # 获取数据
    trainX, testX, trainY, testY, scaler = train_and_test(
        code=code,
        time=time,
        features_names=['open', 'high', 'low', 'preclose', 'volume', 'amount', 'turn', 'pctChg', 'peTTM', 'pbMRQ',
                        'psTTM', 'pcfNcfTTM', 'isST'],
        time_step=time_step,
        train_size=train_size,
    )

    # '''
    # 当 batch_size = len(trainX) 时，
    # train_loader = trainX, trainY
    # test_loader = testX, testY
    # '''
    # batch_size = len(trainX)
    # train_loader, test_loader = data_loader(batch_size, trainX, trainY, testX, testY)

    # 设置参数
    input_dim = len(trainX[0][0])  # 数据的特征数
    output_dim = len(trainY[0][0])  # 预测值的特征数

    # 设置超参数
    hidden_dim = hidden_dim  # 隐藏层的神经元个数
    num_layers = num_layers  # LSTM的层数
    learning_rate = learning_rate  # 学习率
    num_epochs = num_epochs  # 设定数据遍历次数

    # 网络模型
    model = myLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=output_dim,
    )

    # 定义优化器和损失函数
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 使用Adam优化算法
    loss_fn = torch.nn.MSELoss(reduction='mean')  # 使用均方差作为损失函数

    # 打印模型结构
    # print(model)

    # 开始训练
    hist = np.zeros(num_epochs)  # 储存loss
    for epoch in range(num_epochs):
        # Initialise hidden state
        # Don't do this if you want your LSTM to be stateful
        # model.hidden = model.init_hidden()

        # Forward pass
        y_train_pred = model(trainX)

        loss = loss_fn(y_train_pred, trainY)
        if epoch % 10 == 0 and epoch != 0:  # 每训练十次，打印一次均方差
            print("Epoch ", epoch, "MSE: ", loss.item())
        hist[epoch] = loss.item()

        # Zero out gradient, else they will accumulate between epochs 将梯度归零
        optimiser.zero_grad()

        # Backward pass
        loss.backward()

        # Update parameters
        optimiser.step()

    torch.save(model.state_dict(), "{}".format(dir))
    print("Model saved in {}".format(dir))

    # 传出最后一个预测值，即明日的收盘价
    pred = model(testX)
    pred = pred.detach().numpy()[:, -1, 0]
    pred = scaler.inverse_transform(pred.reshape(-1, 1))

    return trainX, testX, trainY, testY, scaler, dir, pred[-1][0]


if __name__ == '__main__':
    # 获取数据
    code = 'sh.600036'
    time = 400
    model_name = code.split('.')[0] + code.split('.')[1]
    time_step = 10
    train_size = 0.8
    hidden_dim = 32
    num_layers = 2
    learning_rate = 0.01
    num_epochs = 200

    trainX, testX, trainY, testY, scaler, dir, pred = train_lstm_for_stock(
        code=code,
        time=time,
        dir='./models/{}.pth'.format(model_name),
        time_step=time_step,
        train_size=train_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
    )

    print(pred)
