# 基于 LSTM 网络的股票预测模型

> 注意：投资须谨慎，本项目预测模型非常简单，对选股投资无实际参考价值

`Python3.9` `Pytorch` `LSTM` `baostock`

## 1. 简介
本项目基于`python3.9`，通过`baostock`模块爬去a股数据，利用`Pytorch`模块搭建LSTM神经网络，用于预测个股收盘价格。

其中 LSTM 网络模型框架为一层`nn.LSTM`，隐藏状态层为`2`。一层全连接层，用于输出股价。

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

### 3. 执行主程序
> 注意：主程序用以测试网络在测试集上的表现，具体预测请运行`prediction.py`脚本

建议初学者使用Pycharm等IDEA打开文件并执行

修改`main.py`文件中的`code_name`变量，填入对应的股票名称后，运行主程序。

运行成功后，在当前目录下会出现`prediction.csv`文件，第一行为股票名称，第二行为预测的结果（即未来最近的一个交易日的收盘价预测值）

### 4. 预测

类似主程序的设置，运行[prediction.py](predition.py)脚本

### 5. 其他
进阶使用，可以自行修改相关参数和网络结构

## 3. 仓库框架
1. 主程序

存储在 [main.py文件](main.py)
2. 数据预处理

封装在 [data_pre.py文件](./data_pre.py)
3. 网络结构

存储在 [lstm_model.py文件](./lstm_model.py)
4. 网络训练源码

封装在 [train.py文件](./train.py)
5. 预测和测试源码

封装在 [test_and_prediction.py文件](test_and_prediction.py)
6. 获取股票基本信息

存储在 [stock_basic.py文件](stock_basic.py)
7. 训练后的网络模型参数

存储在 [models目录](./models) 下，且模型名称为股票代码