import baostock as bs

import torch
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def fill_nan_with_neighbors(arr):
    """
    填充缺失值函数
    :param arr: 一列数据
    :return: 填充后的一列数据
    """
    # 填充缺失值函数
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            prev_value = arr[i - 1] if i > 0 else None
            next_value = arr[i + 1] if i < len(arr) - 1 else None

            # 取前后两个非 NaN 的平均数
            if prev_value is not None and next_value is not None and not np.isnan(prev_value) and not np.isnan(
                    next_value):
                arr[i] = (prev_value + next_value) / 2
            elif prev_value is not None and not np.isnan(prev_value):
                arr[i] = prev_value
            elif next_value is not None and not np.isnan(next_value):
                arr[i] = next_value
    return arr


def get_stock(code: str, time: int) -> pd.DataFrame:
    """
    获取个股信息
    :param code: 股票代码
    :param time: 取天数
    :return: 原始股票信息
    """
    # 登陆系统
    # lg = bs.login()
    # 显示登陆返回信息

    # 获取沪深A股历史K线数据
    # 详细指标参数，参见“历史行情指标参数”章节；“分钟线”参数与“日线”参数不同。“分钟线”不包含指数。
    # 分钟线指标：date,time,code,open,high,low,close,volume,amount,adjustflag
    # 周月线指标：date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg

    stock_code = code
    rs = bs.query_history_k_data_plus(
        code=stock_code,
        fields="date, code, open, high, low, close, preclose, volume, amount, adjustflag, turn, tradestatus, pctChg, peTTM, pbMRQ, psTTM, pcfNcfTTM, isST",
        frequency="d",
        adjustflag="3"
    )

    # 打印结果集
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())

    result = pd.DataFrame(data_list, columns=rs.fields)

    # 登出系统
    # bs.logout()

    return result.tail(time)


def data_pre(origin_data: pd.DataFrame, features_names: list) -> pd.DataFrame:
    """
    分类特征和标签
    :param features_names: 选取特征
    :param origin_data: 原始股票数据
    :return: 完整的数据迭代器
    """

    features = pd.DataFrame()

    # 特征和标签分离 close收盘价作为标签
    for name in features_names:
        features[name] = origin_data[name]

    features = features.values
    labels = origin_data['close'].values

    # 将空字符串替换为 NaN
    features = np.where(features == '', np.nan, features)
    labels = np.where(labels == '', np.nan, labels)

    # 转换 features 和 labels 中的元素为浮点数
    features = features.astype(float)
    labels = labels.astype(float)

    # 对 features 每一列填充 NaN
    for col in range(features.shape[1]):
        features[:, col] = fill_nan_with_neighbors(features[:, col])

    # 对 labels 填充 NaN
    labels = fill_nan_with_neighbors(labels)

    features = pd.DataFrame(features)
    features.columns = features_names
    features.set_index(origin_data['date'], inplace=True)

    labels = pd.DataFrame(labels)
    labels.columns = ['close']
    labels.set_index(origin_data['date'], inplace=True)

    # 标准化features
    scaler = MinMaxScaler(feature_range=(-1, 1))
    for name in features_names:
        features[name] = scaler.fit_transform(features[name].values.reshape(-1, 1))
    labels['close'] = scaler.fit_transform(labels['close'].values.reshape(-1, 1))

    final_data = features.copy()
    final_data['label'] = labels['close'].shift(-1)
    final_data.dropna(how='any', inplace=True)
    final_data = final_data.astype(np.float32)

    return final_data, scaler


def train_and_test(code: str, time: int, features_names, time_step=20, train_size=0.8):
    """
    训练集和测试集（可直接使用）
    :param code: 股票代码
    :param time: 取的天数
    :param features_names: 选取特征
    :param time_step: 时间步
    :param train_size: 划分比例，默认训练集80%
    :return: 返回 trainX, testX, trainY, testY
    """
    origin_data = get_stock(
        code=code,
        time=time,
    )

    final_data, scaler = data_pre(
        origin_data=origin_data,
        features_names=features_names,
    )

    # 创建两个列表，用来存储数据的特征和标签
    data_feat, data_target = [], []

    for index in range(len(final_data) - time_step):
        # 构建特征集
        data_feat.append(final_data[features_names][index: index + time_step].values)
        # 构建target集
        data_target.append(final_data['label'][index:index + time_step])

    # 将特征集和标签集整理成numpy数组
    data_feat = np.array(data_feat)
    data_target = np.array(data_target)

    # 这里按照比例划分训练集和测试集
    test_size = int(np.round((1 - train_size) * final_data.shape[0]))  # np.round(1)是四舍五入
    train_size = data_feat.shape[0] - (test_size)
    # print(test_set_size)  # 输出测试集大小
    # print(train_size)  # 输出训练集大小

    features_num = len(features_names)
    trainX = torch.from_numpy(data_feat[:train_size].reshape(-1, time_step, features_num)).type(torch.Tensor)
    # 这里第一个维度自动确定，我们认为其为batch_size，因为在LSTM类的定义中，设置了batch_first=True
    testX = torch.from_numpy(data_feat[train_size:].reshape(-1, time_step, features_num)).type(torch.Tensor)
    trainY = torch.from_numpy(data_target[:train_size].reshape(-1, time_step, 1)).type(torch.Tensor)
    testY = torch.from_numpy(data_target[train_size:].reshape(-1, time_step, 1)).type(torch.Tensor)

    return trainX, testX, trainY, testY, scaler


def data_loader(batch_size, trainX, trainY, testX, testY):
    """
    数据迭代器
    :param batch_size: 批量大小，数据较小，可以设置为全部
    :param trainX: 训练特征
    :param trainY: 训练标签
    :param testX: 测试特征
    :param testY: 测试标签
    :return: 返回 train_loader, test_loader
    """
    train = TensorDataset(trainX, trainY)
    test = TensorDataset(testX, testY)
    train_loader = DataLoader(dataset=train,
                              batch_size=batch_size,
                              shuffle=False)

    test_loader = DataLoader(dataset=test,
                             batch_size=batch_size,
                             shuffle=False)

    return train_loader, test_loader


if __name__ == '__main__':
    trainX, testX, trainY, testY, scaler = train_and_test(
        code='sh.600036',
        time=400,  # 从今日起前400天
        features_names=['open', 'high', 'low', 'preclose', 'volume', 'amount', 'turn', 'pctChg', 'peTTM', 'pbMRQ',
                        'psTTM', 'pcfNcfTTM', 'isST'],
        time_step=20,
        train_size=0.8
    )

    '''
    当 batch_size = len(trainX) 时，
    train_loader = trainX, trainY
    test_loader = testX, testY
    '''
    batch_size = len(trainX)
    train_loader, test_loader = data_loader(batch_size, trainX, trainY, testX, testY)
