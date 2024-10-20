import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt

from model import LSTMModel
from dataset import get_dataset, get_stock


def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()

    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


stock_code = "sh.600048"
index_code = "sh.000001"

# 1. data
df, mean_and_std, date_and_code = get_stock(stock_code, index_code)

feature, label = get_dataset(df)


# 预测函数，多步预测
def predict_future(model, initial_sequence, steps):
    model.eval()
    predictions = []
    seq = initial_sequence
    for _ in range(steps):
        with torch.no_grad():
            input_seq = torch.tensor(seq[-time_step:], dtype=torch.float32).unsqueeze(0).to(device)
            pred = model(input_seq).item()
            predictions.append(pred)
            seq = np.vstack([seq, np.zeros((1, seq.shape[1]))])  # 加入新时间步
            seq[-1, -1] = pred  # 用预测值替换最新的label
    return predictions


# 加载模型
input_size = len(feature[1])  # features numbers
hidden_size = 64
num_layers = 3
output_size = 1
time_step = 10
device = try_gpu()

prediction_step = 30

model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
model.load_state_dict(torch.load('./model_param/lstm_model.pth'))
model.eval()

initial_sequence = feature[-time_step:]
predicted_values = predict_future(model, initial_sequence, steps=prediction_step)

mean_df, std_df = mean_and_std
mean = mean_df["close"][0]
std = std_df["close"][0]

# print(predicted_values, len(predicted_values))
prediction_data = ((label + mean) * std)[-1].tolist()[0] + (predicted_values + mean) * std
df_prediction_data = pd.DataFrame(prediction_data)
df_prediction_data.to_csv("./prediction_data/pre_30steps.csv".format(prediction_step))

# 绘图
plt.figure(figsize=(10, 6))
# 真实数据
plt.plot(range(len(label)), (label + mean) * std, label='previous', color='blue')
# 预测数据
plt.plot(range(len(label), len(label) + len(predicted_values)), (predicted_values + mean) * std, label='prediction',
         color='red')

plt.legend()
plt.xlabel('time')
plt.ylabel('value')
plt.title('previous and prediction')
plt.savefig('./prediction_plot/pre_{}steps.png'.format(prediction_step))
plt.show()
