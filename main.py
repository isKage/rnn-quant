import baostock as bs
import pandas as pd

from stock_basic import stock_name, stock_code
from test_and_prediction import test_and_prediction

bs.login()

code_name = ['招商银行', '比亚迪', '东方财富', '宁德时代', '贵州茅台', '药明康德', '珀莱雅', '万达电影', '中国石化']
code = [stock_code(name) for name in code_name]
# print(code_name)
# print(code)

prediction = []
# 获取数据
for c in code:
    time = 400
    model_name = c.split('.')[0] + c.split('.')[1]
    time_step = 10
    train_size = 0.8
    hidden_dim = 32
    num_layers = 2
    learning_rate = 0.01
    num_epochs = 200

    model_dir, pred = test_and_prediction(
        code=c,
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
    prediction.append(pred)
    # print(model_dir)
    # print(pred)

result = pd.DataFrame(prediction).transpose()
result.columns = code_name
result.to_csv('prediction.csv')

bs.logout()
