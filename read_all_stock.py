import baostock as bs
import pandas as pd

from predition import run
from lstm_model import *
from stock_basic import stock_code


def sz50():
    # 获取上证 50 成分股
    rs = bs.query_sz50_stocks()
    print('query_sz50 error_code:' + rs.error_code)
    print('query_sz50  error_msg:' + rs.error_msg)

    # 打印结果集
    sz50_stocks = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        sz50_stocks.append(rs.get_row_data())
    result = pd.DataFrame(sz50_stocks, columns=rs.fields)

    return result


if __name__ == '__main__':
    bs.login()

    sz50_stocks = sz50()
    # code = 'sh.600036'
    stock_name = sz50_stocks['code_name'].to_list()
    stock_code = sz50_stocks['code'].to_list()

    prediction = []
    latest = []
    rate = []
    # 获取数据
    for c in stock_code:
        time = 100
        true, pred, loss = run(c, time)
        true = list(map(float, true))

        prediction.append(pred)
        latest.append(true[-1])
        rate.append((pred / true[-1]) - 1)

    result = pd.DataFrame([latest, prediction, rate])
    result.index = ['latest', 'prediction', 'rate']
    result.columns = stock_name
    result.to_csv('test.csv')

    bs.logout()

