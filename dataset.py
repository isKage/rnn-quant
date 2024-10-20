import baostock as bs
import pandas as pd
import numpy as np

import torch
from torch.utils.data import TensorDataset


class StockBasis:
    def __init__(self, code):
        self.code = code  # 股票代码

    def get_stock_day_data(self, code=None):
        """
        获取单只股票或指数的每日历史数据
        :param code: 股票代码或指数代码，默认为实例的 code
        :return: DataFrame 格式的历史数据
        """
        if code is None:
            code = self.code  # 如果没有传入代码参数，使用初始化时的股票代码

        rs = bs.query_history_k_data_plus(
            code=code,
            start_date="2016-01-01",
            fields="date,code,open,high,low,close,preclose,volume,amount,turn,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ",
            frequency="d",
            adjustflag="3"
        )

        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())  # 将每条记录加入列表
        result = pd.DataFrame(data_list, columns=rs.fields)

        return result

    def add_index_day_data(self, index_code="sh.000001"):
        """
        将指数数据加入到股票数据中，以便进行对比分析
        :param index_code: 指数代码
        :return: 合并后的 DataFrame
        """
        # 获取股票数据
        stock_data = self.get_stock_day_data()

        # 获取指数数据
        index_data = self.get_stock_day_data(index_code)
        index_data = index_data.drop(columns=['code'])
        index_data = index_data.drop(columns=['peTTM', 'psTTM', 'pcfNcfTTM', 'pbMRQ'])

        # 将两个 DataFrame 进行合并，按照日期对齐
        merged_data = pd.merge(stock_data, index_data, on='date', suffixes=('', '_index'))

        # 处理除 'code' 和 'date' 以外的列
        columns_to_convert = [col for col in merged_data.columns if col not in ['code', 'date']]

        # 将所有列的空字符替换为 NaN，并转为 float 类型
        merged_data[columns_to_convert] = merged_data[columns_to_convert].replace('', np.nan).astype(float)

        # 对 NaN 值进行填充，使用上下行均值
        for col in columns_to_convert:
            # 使用滚动窗口求前后均值填充
            merged_data[col] = merged_data[col].fillna(
                merged_data[col].interpolate(method='linear', limit_direction='both'))

        return merged_data


def get_stock(stock_code, index_code):
    # 初始化 baostock
    bs.login()

    # 实例化 StockBasis 类，传入股票代码
    stock_basis = StockBasis(stock_code)

    # 获取股票和指数合并数据
    df = stock_basis.add_index_day_data(index_code)  # 传入指数代码
    # print(df.shape)
    # print("合并后的数据：\n", merged_df.head())

    # 退出登录
    bs.logout()

    columns_name = df.columns
    columns_name = columns_name.to_list()
    columns_name = columns_name[2:]
    date = df['date'].to_list()
    code = df['code'][0]

    df.drop(columns=['date', 'code'], inplace=True)

    mean = []
    std = []
    for name in columns_name:
        mean.append(df[name].mean())
        std.append(df[name].std())

    mean_df = pd.DataFrame([mean], columns=columns_name)
    std_df = pd.DataFrame([std], columns=columns_name)

    for name in columns_name:
        df[name] = (df[name] - mean_df[name][0]) / std_df[name][0]

    return df, (mean_df, std_df), (date, code)


def get_dataset(df):
    labels = df['close'].values
    features = df.drop(columns=['close']).values

    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    return features_tensor, labels_tensor


def create_sequences(feature, label, time_step):
    X, Y = [], []
    for i in range(len(feature) - time_step):
        X.append(feature[i:i + time_step])
        Y.append(label[i + time_step])
    return np.array(X), np.array(Y)


if __name__ == "__main__":
    stock_code = "sh.600048"
    index_code = "sh.000001"

    df, mean_and_std, date_and_code = get_stock(stock_code, index_code)

    dataset = get_dataset(df)

    # print(dataset)
    # print(mean_and_std)
    # print(date_and_code)
