import matplotlib.pyplot as plt

from lstm_model import *

from train import train_lstm_for_stock


def test_and_prediction(
        code: str,
        model_name: str,
        fig: int = 0,
        time=400,
        time_step=10,
        train_size=0.8,
        hidden_dim=32,
        num_layers=2,
        learning_rate=0.01,
        num_epochs=200,
):
    """
    训练、预测、测试三合一
    :param code: 股票代码
    :param model_name: 模型名称
    :param fig: 是否绘图
    :param time: 距今时间
    :param time_step: 时间步
    :param train_size: 训练集划分
    :param hidden_dim: 隐藏层的神经元个数
    :param num_layers: LSTM的层数
    :param learning_rate: 学习率
    :param num_epochs: 设定数据遍历次数
    :return: (模型地址, 预测值 单值预测)
    """
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

    # 设置参数
    input_dim = len(trainX[0][0])  # 数据的特征数
    output_dim = len(trainY[0][0])  # 预测值的特征数

    model = myLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=output_dim,
    )

    model_param_dict = torch.load("./models/{}.pth".format(model_name))  # parameters' dict
    model.load_state_dict(model_param_dict)

    if fig == 1:
        """训练集效果图"""
        # 无论是真实值，还是模型的输出值，它们的维度均为（batch_size, seq, 1），seq=20
        # 我们的目的是用前20天的数据预测今天的股价，所以我们只需要每个数据序列中第20天的标签即可
        # 因为前面用了使用DataFrame中shift方法，所以第20天的标签，实际上就是第21天的股价
        # 计算训练得到的模型在训练集上的均方差
        train_pred_value = model(trainX)

        train_pred_value = train_pred_value.detach().numpy()[:, -1, 0]
        train_true_value = trainY.detach().numpy()[:, -1, 0]  # 纵坐标还有负的，因为前面进行缩放，现在让数据还原成原来的大小

        train_pred_value = scaler.inverse_transform(train_pred_value.reshape(-1, 1))
        train_true_value = scaler.inverse_transform(train_true_value.reshape(-1, 1))

        plt.plot(train_pred_value, label="Preds")  # 预测值
        plt.plot(train_true_value, label="Data")  # 真实值
        plt.title('train prediction')
        plt.legend()
        plt.show()

        """测试集效果图"""
        test_pred_value = model(testX)
        test_pred_value = test_pred_value.detach().numpy()[:, -1, 0]
        test_true_value = testY.detach().numpy()[:, -1, 0]

        test_pred_value = scaler.inverse_transform(test_pred_value.reshape(-1, 1))
        test_true_value = scaler.inverse_transform(test_true_value.reshape(-1, 1))

        plt.plot(test_pred_value, label="Preds")  # 预测值
        plt.plot(test_true_value, label="Data")  # 真实值
        plt.title('test prediction')
        plt.legend()
        plt.show()

    return dir, pred


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

    model_dir, pred = test_and_prediction(
        code=code,
        model_name=model_name,
        fig=0,
        time=time,
        time_step=time_step,
        train_size=train_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
    )

    print(model_dir)
    print(pred)
