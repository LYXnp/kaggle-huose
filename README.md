	1项目介绍
基于kaggle网站所提供的爱荷华州埃姆斯的住宅数据信息，预测每间房屋的销售价格，数据的标签SalePrice是连续性数据，因此可以判定这是一个回归问题。
最终目标：预测每一个房屋的销售价格。对于测试集中的每个ID，预测SalePrice变量的值
判定标准：根据预测值的对数与观察到的销售价格的对数之间的均方根误差（RMSE）评估提交的内容（采取对数意味着预测昂贵房屋和廉价房屋的错误将同等影响结果).
数据描述：数据来源于kaggle网站。数据分为训练数据集和测试数据集。两个数据集都包括每栋房⼦的特征，如街道类型、建造年份、房顶类型、地下室状况等特征值。这些特征值有连续的数字、离散的标签甚⾄是缺失值“na”。只有训练数据集包括了每栋房⼦的价格，也就是标签。训练数据和测试数据分别各有1460条，数据的特征列有79个，期中35个是数值类型的，44个类别类型
2模型代码分布及作用介绍
2.1获取和读取数据集
2.1.1加载依赖库（torch）：
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
import d2l as d2l
print(torch.__version__)
torch.set_default_tensor_type(torch.FloatTensor)
2.1.2加载数据：
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
2.1.3查看一下训练集，测试集的特征（维度）：
print(train_data.shape)
print(test_data.shape)
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
结果：
(1460, 81)
(1459, 80)
   Id  MSSubClass MSZoning  LotFrontage SaleType SaleCondition  SalePrice
0   1          60       RL         65.0       WD        Normal     208500
1   2          20       RL         80.0       WD        Normal     181500
2   3          60       RL         68.0       WD        Normal     223500
3   4          70       RL         60.0       WD       Abnorml     140000
2.1.4去除特征id，提高训练模型的准确度（第⼀个特征是Id，它能帮助模型记住每个训练样本，但难以推⼴到测试样本）：
all_features= pd.concat((train_data.iloc[:, 1:-1],test_data.iloc[:, 1:]))

2.2预处理数据
2.2.1对连续数值的特征做标准化
numeric_features = all_features.dtypes[all_features.dtypes !='object'].index
all_features[numeric_features] =all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
# 标准化后，每个特征的均值变为0，所以可以直接⽤0来替换缺失值
all_features = all_features.fillna(0)
2.2.2将离散数值转成指示特征
# dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features.shape # (2919, 354)
2.2.3通过 values 属性得到NumPy格式的数据，并转成 ndarray ⽅便后⾯的训练
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values.astype(np.float32))
test_features = torch.tensor(all_features[n_train:].values.astype(np.float32))
train_labels = torch.tensor(train_data.SalePrice.values.astype(np.float32)).view(-1, 1)

2.3训练模型
2.3.1使⽤⼀个基本的线性回归模型和平⽅损失函数来训练模型
loss = torch.nn.MSELoss()
def get_net(feature_num):
    net = nn.Linear(feature_num, 1)
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net
2.3.2使用对数均⽅根误差评价模型
def log_rmse(net, features, labels):
 with torch.no_grad():
 # 将⼩于1的值设成1，使得取对数时数值更稳定
 clipped_preds = torch.max(net(features), torch.tensor(1.0))
 rmse = torch.sqrt(2 * loss(clipped_preds.log(),
labels.log()).mean())
 return rmse.item()
2.3.3训练函数（Adam优化算法）
def train(net, train_features, train_labels, test_features,
test_labels,
 num_epochs, learning_rate, weight_decay, batch_size):
 train_ls, test_ls = [], []
 dataset = torch.utils.data.TensorDataset(train_features,
train_labels)
 train_iter = torch.utils.data.DataLoader(dataset, batch_size,
shuffle=True)
 # 这⾥使⽤了Adam优化算法
 optimizer = torch.optim.Adam(params=net.parameters(),
lr=learning_rate, weight_decay=weight_decay)
 net = net.float()
 for epoch in range(num_epochs):
 for X, y in train_iter:
 l = loss(net(X.float()), y.float())
 optimizer.zero_grad()
 l.backward()
 optimizer.step()
 train_ls.append(log_rmse(net, train_features,
train_labels))
 if test_labels is not None:
 test_ls.append(log_rmse(net, test_features,
test_labels))
return train_ls, test_ls
2.4K折交叉验证
2.4.1它返回第i折交叉验证时所需要的训练和验证数据
def get_k_fold_data(k, i, X, y):
 # 返回第i折交叉验证时所需要的训练和验证数据
 assert k > 1
 fold_size = X.shape[0] // k
 X_train, y_train = None, None
 for j in range(k):
 idx = slice(j * fold_size, (j + 1) * fold_size)
 X_part, y_part = X[idx, :], y[idx]
 if j == i:
 X_valid, y_valid = X_part, y_part
 elif X_train is None:
 X_train, y_train = X_part, y_part
 else:
 X_train = torch.cat((X_train, X_part), dim=0)
 y_train = torch.cat((y_train, y_part), dim=0)
 return X_train, y_train, X_valid, y_valid
2.4.2训练k次并返回训练和验证的平均误差
def k_fold(k, X_train, y_train, num_epochs,
 learning_rate, weight_decay, batch_size):
 train_l_sum, valid_l_sum = 0, 0
 for i in range(k):
 data = get_k_fold_data(k, i, X_train, y_train)
 net = get_net(X_train.shape[1])
 train_ls, valid_ls = train(net, *data, num_epochs,
learning_rate,
 weight_decay, batch_size)
 train_l_sum += train_ls[-1]
 valid_l_sum += valid_ls[-1]
 if i == 0:
 d2l.semilogy(range(1, num_epochs + 1), train_ls,
'epochs', 'rmse',
 range(1, num_epochs + 1), valid_ls,
 ['train', 'valid'])
 print('fold %d, train rmse %f, valid rmse %f' % (i,
train_ls[-1], valid_ls[-1]))
 return train_l_sum / k, valid_l_sum / k
输出：
 
fold 0, train rmse 0.241054, valid rmse 0.221462 
fold 1, train rmse 0.229857, valid rmse 0.268489 
fold 2, train rmse 0.231413, valid rmse 0.238157 
fold 3, train rmse 0.237733, valid rmse 0.218747 
fold 4, train rmse 0.230720, valid rmse 0.258712 
5-fold validation: avg train rmse 0.234155, avg valid rmse 0.241113

2.5模型选择
2.5.1调参
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels,
num_epochs, lr, weight_decay, batch_size)
print('%d-fold validation: avg train rmse %f, avg valid rmse %f' %
(k, train_l, valid_l))

2.6预测
2.6.1预测函数
def train_and_pred(train_features, test_features, train_labels,
test_data,
 num_epochs, lr, weight_decay, batch_size):
 net = get_net(train_features.shape[1])
 train_ls, _ = train(net, train_features, train_labels, None,
None,
 num_epochs, lr, weight_decay, batch_size)
 d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs',
'rmse')
 print('train rmse %f' % train_ls[-1])
 preds = net(test_features).detach().numpy()
 test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
 submission = pd.concat([test_data['Id'],
test_data['SalePrice']], axis=1)
 submission.to_csv('./submission.csv', index=False)

train_and_pred(train_features, test_features, train_labels,
test_data, num_epochs, lr, weight_decay, batch_size)
结果：
 
train rmse 0.229943
3测试调参和结果分析（全部采用五折验证表示调参结果，第一折结果用于画图）
3.1测试2种loss函数
3.1.1平滑L1损失函数
torch.nn.SmoothL1Loss(): 
平滑L1损失函数，与均方误差损失函数类似，但具有较小的梯度和较大的鲁棒性
 
fold 0, train rmse 0.146054, valid rmse 0.135054
fold 1, train rmse 0.139658, valid rmse 0.161704
fold 2, train rmse 0.140050, valid rmse 0.150213
fold 3, train rmse 0.145135, valid rmse 0.131494
fold 4, train rmse 0.137347, valid rmse 0.163279
5-fold validation: avg train rmse 0.141649, avg valid rmse 0.148349
3.1.2均方误差损失函数
torch.nn.MSELoss(): 均方误差损失函数，常用于回归问题。

 
fold 0, train rmse 0.240366, valid rmse 0.222322
fold 1, train rmse 0.229746, valid rmse 0.267980
fold 2, train rmse 0.231853, valid rmse 0.238125
fold 3, train rmse 0.238350, valid rmse 0.218955
fold 4, train rmse 0.230894, valid rmse 0.259115
5-fold validation: avg train rmse 0.234242, avg valid rmse 0.241299
3.2调整learning rate的变化
3.2.1learning rate=10
 
fold 0, train rmse 0.211058, valid rmse 0.209903
fold 1, train rmse 0.205689, valid rmse 0.229507
fold 2, train rmse 0.202462, valid rmse 0.210207
fold 3, train rmse 0.209855, valid rmse 0.209073
fold 4, train rmse 0.202039, valid rmse 0.245910
5-fold validation: avg train rmse 0.206221, avg valid rmse 0.220920
3.2.2 learning rate=20
 
fold 0, train rmse 0.187535, valid rmse 0.204418
fold 1, train rmse 0.183006, valid rmse 0.206825
fold 2, train rmse 0.181062, valid rmse 0.200134
fold 3, train rmse 0.189394, valid rmse 0.198404
fold 4, train rmse 0.178531, valid rmse 0.237940
5-fold validation: avg train rmse 0.183906, avg valid rmse 0.209544
3.3测试2种优化算法
3.3.1Adadelta优化算法
Adadelta的主要思想是使用历史梯度平方的加权平均值来调整学习率。具体来说，对于每个参数，Adadelta维护两个状态变量：RMS(平均平方)和delta(更新量)。RMS是历史梯度平方的加权平均值，delta是最近的权重更新的加权平均值。
 
fold 0, train rmse 5.577818, valid rmse 5.595828
fold 1, train rmse 5.579638, valid rmse 5.595270
fold 2, train rmse 5.568633, valid rmse 5.595203
fold 3, train rmse 5.587885, valid rmse 5.528215
fold 4, train rmse 5.569706, valid rmse 5.581266
5-fold validation: avg train rmse 5.576736, avg valid rmse 5.579156
3.3.2 RMSprop优化算法
RMSprop的主要思想是使用指数加权移动平均（Exponential Moving Average，EMA）来调整学习率。具体来说，对于每个参数，RMSprop维护一个状态变量RMS（Root Mean Square）来表示历史梯度平方的移动平均值。
 
fold 0, train rmse 0.211836, valid rmse 0.209433
fold 1, train rmse 0.205962, valid rmse 0.227017
fold 2, train rmse 0.203293, valid rmse 0.213957
fold 3, train rmse 0.210920, valid rmse 0.204404
fold 4, train rmse 0.200903, valid rmse 0.243299
5-fold validation: avg train rmse 0.206583, avg valid rmse 0.219622
3.4测试2种初始化方法
3.4.1 Min-max标准化
Min-max标准化是将数值型特征缩放到[0,1]之间的范围内

 
fold 0, train rmse 0.365775, valid rmse 0.336179
fold 1, train rmse 0.358029, valid rmse 0.383553
fold 2, train rmse 0.357377, valid rmse 0.382297
fold 3, train rmse 0.365971, valid rmse 0.363139
fold 4, train rmse 0.360295, valid rmse 0.360968
5-fold validation: avg train rmse 0.361489, avg valid rmse 0.365227
3.4.2等距标准化
等距标准化是将数值型特征缩放到[-1,1]之间的范围内
 
fold 0, train rmse 0.373244, valid rmse 0.347196
fold 1, train rmse 0.366587, valid rmse 0.391712
fold 2, train rmse 0.366596, valid rmse 0.391954
fold 3, train rmse 0.374129, valid rmse 0.367569
fold 4, train rmse 0.367627, valid rmse 0.375378
5-fold validation: avg train rmse 0.369637, avg valid rmse 0.374762
3.5测试2种epoch num
3.5.1epoch num=100
上面结果全是epoch num=100时的结果（参考3.1.2）

3.5.2epoch num=200
对比于3.1.2
 
fold 0, train rmse 0.204717, valid rmse 0.205854
fold 1, train rmse 0.200034, valid rmse 0.221393
fold 2, train rmse 0.196835, valid rmse 0.206112
fold 3, train rmse 0.204121, valid rmse 0.203564
fold 4, train rmse 0.195172, valid rmse 0.243526
5-fold validation: avg train rmse 0.200176, avg valid rmse 0.216090
3.6测试2种k折交叉验证的k值
3.6.1  k=4
 
fold 0, train rmse 0.211201, valid rmse 0.201209
fold 1, train rmse 0.194141, valid rmse 0.245456
fold 2, train rmse 0.210237, valid rmse 0.186490
fold 3, train rmse 0.199161, valid rmse 0.241473
4-fold validation: avg train rmse 0.203685, avg valid rmse 0.218657

3.6.2  k=6
 
fold 0, train rmse 0.200983, valid rmse 0.209275
fold 1, train rmse 0.204514, valid rmse 0.194379
fold 2, train rmse 0.190930, valid rmse 0.242927
fold 3, train rmse 0.203476, valid rmse 0.177570
fold 4, train rmse 0.199956, valid rmse 0.216875
fold 5, train rmse 0.199268, valid rmse 0.241877
6-fold validation: avg train rmse 0.199855, avg valid rmse 0.213817
3.7测试对比不做数据预处理和做了预处理的效果差异
3.7.1不做预处理
 
 
fold 0, train rmse 0.198853, valid rmse 0.212584
fold 1, train rmse 0.216752, valid rmse 1.654956
fold 2, train rmse 0.296894, valid rmse 0.333365
fold 3, train rmse 0.210439, valid rmse 0.205812
fold 4, train rmse 0.247431, valid rmse 0.281738
fold 5, train rmse 0.204136, valid rmse 0.256740
6-fold validation: avg train rmse 0.229084, avg valid rmse 0.490866
3.7.2做预处理
与结果3.6.2做对比
3.8分析结果
(1)	平滑L-1损失函数较均方误差损失函数，训练的模型，误差相对较小。
(2)	而学习率（learning-rate）的改变，相对于模型的优劣性没有较大的变化自适应学习率优化算法addelta，对于本问题，没有较大的优化作用，反而admp和RMSProp效果更好。
(3)	用于调参标准化的最大最小值标准化（min-max）和等距标准化的测试效果没有标准差标准化（Z-score）的测试效果较好。
(4)	对于epoch，越大越好，但存在一定的上限，会导致梯度不再下降，从而终止模型的训练。
(5)	本次实验对数据预处理采用不同的归一化手段，而未做数据预处理，会导致网络模型收敛速度慢，或者根本就没有办法收敛，同时表现不稳定，特征丢失严重的情况发生，所以在使用机器学习、深度学习、神经网络等算法之前，通常需要对数据进行预处理，包括归一化、标准化、缺失值处理等，以提高模型训练的效果和稳定性。

4遇到的问题和解决方法：
d2lzh_pytorch的报错：
原因：没有安装这个模块
解决方法：把d2lzh_pytorch放入项目文件中就可以识别加载，但是还要安装torchtext，tqdp（已经安装过）
Pip install torchtext
Pytorch警告：
原因：Urllib和chardet版本不相容
解决方法：卸载更新安装requests
pip uninstall urllib3
pip uninstall chardet
pip install --upgrade requests
数据类型报错：
报错code：
train_features=torch.tensor(all_features[:n_train].values,dtype=torch.float)

原因：数组中存在不支持的数据类型 numpy.object_
解决方法：
train_features=torch.tensor(all_features[:n_train].values.astype(np.float32))


