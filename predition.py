import baostock as bs
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from data_pre import get_stock, data_pre, train_and_test

from lstm_model import *
from stock_basic import stock_code


def run(code: str, time: int = 400):
    # pred = model(torch.randn(1, 1, input_dim))
    stock = get_stock(code, time)

    features_name = ['open', 'high', 'low', 'preclose', 'volume', 'amount', 'turn', 'pctChg', 'peTTM', 'pbMRQ', 'psTTM',
                     'pcfNcfTTM', 'isST']
    trainX, testX, trainY, testY, scaler = train_and_test(
        code=code,
        time=time,
        features_names=features_name,
        time_step=20,
        train_size=1,
    )

    hidden_dim = 32
    num_layers = 2
    input_dim = len(features_name)  # 数据的特征数
    output_dim = 1  # 预测值的特征数
    learning_rate = 0.01  # 学习率
    num_epochs = 200  # 设定数据遍历次数

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

    '''
    torch.save(model.state_dict(), "{}".format(dir))
    print("Model saved in {}".format(dir))
    '''

    # 传出最后一个预测值，即明日的收盘价
    pred = model(trainX)
    pred = pred.detach().numpy()[:, -1, 0]
    pred = scaler.inverse_transform(pred.reshape(-1, 1))

    # print(hist.shape)
    # print(type(hist))
    close_list = stock['close']
    # print("close length: {}".format(len(close_list)))
    # print("prediction length: {}".format(pred.shape))
    # print(type(pred))
    # print(close_list)
    # print(pred)
    # print(pred[-1][0])

    # plt.plot(hist)
    # plt.xlabel('loss')
    # plt.ylabel('epochs')
    # plt.show()
    return close_list, pred[-1][0], hist


if __name__ == '__main__':
    bs.login()

    # code = 'sh.600036'
    stock_name = ['招商银行', '比亚迪', '东方财富', '宁德时代', '贵州茅台', '药明康德', '珀莱雅', '万达电影',
                  '中国石化']
    code = [stock_code(name) for name in stock_name]

    prediction = []
    latest = []
    rate = []
    # 获取数据
    for c in code:
        time = 400
        true, pred, loss = run(c, time)
        true = list(map(float, true))

        prediction.append(pred)
        latest.append(true[-1])
        rate.append((pred / true[-1]) - 1)

    result = pd.DataFrame([latest, prediction, rate])
    result.index = ['latest', 'prediction', 'rate']
    result.columns = stock_name
    result.to_csv('prediction.csv')

    '''
    # 创建折线图
    plt.plot(range(len(true)), true, marker='o', label='real close price')
    plt.plot([len(true) - 1, len(true)], [true[-1], pred], 'r--', label='prediction')
    plt.plot(len(true), pred, 'ro', label='prediction')
    # 图表美化
    plt.title('close price of {}'.format(code))
    plt.xlabel('date')
    plt.ylabel('close price')
    plt.legend()
    # 显示图表
    plt.show()

    plt.plot(loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss on train set')
    plt.show()
    '''

    bs.logout()
