import matplotlib
import matplotlib.pyplot as plt
import numpy
import torch
import pandas
import os
from tqdm import tqdm

#选择加速设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取csv数据
train_data = pandas.read_csv("train.csv")
test_data = pandas.read_csv("test.csv")

# 把去掉id的数据拼在一起
# 去掉id的目的是为了防止模型通过记住编号得到对应房价。
all_features = pandas.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
print("train_data.shape:", train_data.shape)
print("test_data.shape:", test_data.shape)
print("all_features:", all_features.shape)
print(train_data.iloc[:5, :8])

# 提取全是数字的特征名字
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index

# 对数据做标准化处理,对应位置赋值
#归一化
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))

# 在标准化数据之后，将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# `Dummy_na=True` 将“na”（缺失值）视为有效的特征值，并为其创建指示符特征。
# pandas.get_dummies把特征为类别值或离散值分成每一个特征为一个类别。
all_features = pandas.get_dummies(all_features, dummy_na=True)
print("all_features.shape:", all_features.shape)

# 把数据分成训练数据和测试数据
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values.astype(numpy.float32))
test_features = torch.tensor(all_features[n_train:].values.astype(numpy.float32))
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)
print("train_features.shape:", train_features.shape)
print("train_features.shape:", test_features.shape)
print("train_labels:", train_labels.shape)

# 数据分批
batch_size = 32
dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_loader = torch.utils.data.DataLoader(dataset,  # 数据
                                           batch_size=batch_size,  # 每个batch大小
                                           shuffle=True,  # 是否打乱数据
                                           num_workers=0,  # 工作线程
                                           pin_memory=True)
print(f"每一批{len(next(iter(train_loader))[0])}个，一共{len(train_loader)}批")


# 定义网络模型
class MyNet(torch.nn.Module):
    def __init__(self, in_put, hidden, hidden1, out_put):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_put, hidden)
        self.linear2 = torch.nn.Linear(hidden, hidden1)
        self.linear3 = torch.nn.Linear(hidden1, out_put)

    def forward(self, data):
        x = self.linear1(data)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        x = self.linear3(x)
        return x


# 取出输入特征个数
in_features = train_features.shape[1]
hidden, hidden1, out_put = 200, 100, 1
model = MyNet(in_features, hidden, hidden1, out_put).to(device)

# 损失函数 loss(xi,yi)=(xi−yi)2
loss = torch.nn.MSELoss()

# 梯度优化算法
learn_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), learn_rate)

print("in_features:", in_features)
print("in_features:", train_features.shape)
print(model)



epochs = 200


def train(train_loader):
    train_ls = []

    for epoch in tqdm(range(epochs)):

        loss_sum = 0
        for train_batch, labels_batch in train_loader:
            train_batch, labels_batch = train_batch.to(device), labels_batch.to(device)
            # preds = torch.clamp(model(train_batch), 1, float('inf'))
            # l = loss(torch.log(preds), torch.log(labels_batch))
            l = loss(model(train_batch), labels_batch)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            loss_sum += l.item()
        train_ls.append(loss_sum)
    plt.plot(range(epochs), train_ls)
    plt.show()


train(train_loader)


def test(test_features):
    test_features = test_features.to(device)
    preds = model(test_features).detach().to("cpu").numpy()
    print(preds.squeeze().shape)

    # pandas.Series 创建新维度
    test_data['SalePrice'] = pandas.Series(preds.squeeze())

    # axis选择拼接的维度
    return pandas.concat([test_data['Id'], test_data['SalePrice']], axis=1)


submission = test(test_features)
submission.to_csv('submission.csv', index=False)
