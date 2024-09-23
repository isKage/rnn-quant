import pandas as pd
import baostock as bs

from read_all_stock import sz50


def determine_sign(row):
    if row['rate'] > 0 and row['real_rate'] > 0:
        return 1
    elif row['rate'] < 0 and row['real_rate'] < 0:
        return 1
    elif row['rate'] > 0 and row['real_rate'] < 0:
        return -1
    elif row['rate'] < 0 and row['real_rate'] > 0:
        return -1
    else:
        return 0


bs.login()

sz50_stocks = sz50()

real_price = []
for code in sz50_stocks['code']:
    rs = bs.query_history_k_data_plus(
        code=code,
        fields="date, code, open, high, low, close",
        start_date="2024-09-19",
        frequency="d",
        adjustflag="3"
    )

    # 打印结果集
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())

    result = pd.DataFrame(data_list, columns=rs.fields)
    result = result.tail(1)
    real_price.append(result['close'].to_list()[0])

real_price = [float(p) for p in real_price]

df = pd.read_csv('test.csv', index_col=0)

df = df.transpose()

df['real_price'] = real_price
df['real_price'] = real_price
df['real_rate'] = df['real_price'] / df['latest'] - 1.0

df['accurate'] = df.apply(determine_sign, axis=1)

count_1 = (df['accurate'] == 1).sum()
count_neg1 = (df['accurate'] == -1).sum()

# 打印结果
print(f"1的个数: {count_1}")
print(f"-1的个数: {count_neg1}")

print("准确率: {}".format(count_1 / (count_1 + count_neg1)))

df.to_csv('accurate.csv', index=True)

bs.logout()
