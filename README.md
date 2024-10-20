# 基于 LSTM 网络的股票预测模型

> 原项目逻辑混乱，故覆写了一份

> 注意：投资须谨慎，本项目预测模型非常简单，对选股投资无实际参考价值

`Python3.9` `Pytorch` `LSTM` `baostock`

## 1. 简介
本项目基于`python3.9`，通过`baostock`模块爬去a股数据，利用`Pytorch`模块搭建LSTM神经网络，用于预测个股收盘价格。

其中 LSTM 网络模型框架为一层`nn.LSTM`，隐藏状态层为`3`(可自主设计)。一层全连接层，用于输出股价。

## 2. 使用方法
### 1. 安装
打开终端，进入一个空目录（用于存放本项目代码），例如
```
mkdir <新目录>
cd <新目录>
```
输入以下指令克隆仓库
```
git clone git@github.com:isKage/rnn-quant.git
```

### 2. 安装必要的包和模块
进入项目根目录
```
cd nn_quant
```
可以列举以下当前目录下的文件，查看是否有[requirements.txt](requirements.txt)文件。执行命令
```
pip install -r requirements.txt
```
> 或者使用conda或是virtualenv创建虚拟环境后执行

### 3. 执行训练脚本

建议初学者使用Pycharm等IDEA打开文件并执行

1. 修改`train.py`文件中的`stock_code`变量，填入对应的股票代码

2. 修改`train.py`文件中的`index_code`变量，填入对应的市场指数代码

3. 运行`train.py`运行成功后，在`model_param`文件夹下会存储对应的模型参数

### 4. 预测

类似训练脚本的设置，

1. 修改`prediction.py`文件中的`stock_code`变量，填入对应的股票代码

2. 修改`prediction.py`文件中的`index_code`变量，填入对应的市场指数代码

3. 运行`prediction.py`运行成功后，在`prediction_data`和`prediction_plot`文件夹下会存储对应的预测数据和图像

数据为`.csv`格式，第一个数据为最新收盘价，往后数据为预测收盘价（如需改变预测天数，可以修改`prediction.py`的`prediction_step`参数

### 5. 其他
进阶使用，可以自行修改相关参数和网络结构

## 3. 仓库框架
1. 训练脚本

存储在 [train.py文件](train.py)

2. 数据预处理

封装在 [dataset.py文件](dataset.py)

3. 网络结构

存储在 [model.py文件](model.py)

4. 训练后的网络模型参数

存储在 [model_param目录](./model_param) 

5. 预测结果和图像

存储在 [prediction_data目录](./prediction_data) 和[prediction_plot目录](prediction_plot)
