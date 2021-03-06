# In this script we play with the dataloader system and a very simple siamese
# network. The goal is to learn how to use the data loading facilities and
# to see whether I'm using wieght sharing in the right way.
# TODO : finish the script
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# Create a fake dataset consisiting of two inputs
np.random.seed(10)


class MyDataset(Dataset):
    def __init__(self):
        self.x1 = torch.randn(1024, 20)
        self.x2 = torch.randn(1024, 20)
        self.y = torch.from_numpy(np.random.randint(0, 2, 1000))

    def __len__(self):
        return(len(self.y))

    def __getitem__(self, idx):
        return [self.x1[idx], self.x2[idx]], self.y[idx]


my_dataset = MyDataset()
dataloader = DataLoader(my_dataset, batch_size=64, shuffle=True, num_workers=2)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.shared_dense = nn.Linear(in_features=20, out_features=10)
        self.dense = nn.Linear(20, 2)

    def forward(self, x):
        x1 = F.relu(self.shared_dense(x[0]))
        x2 = F.relu(self.shared_dense(x[0]))
        x3 = torch.cat([x1, x2], dim=1)
        x3 = F.relu(self.dense(x3))
        return x3


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9)

for epoch in range(10):
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs = [Variable(inputs[0]), Variable(inputs[1])]
        labels = Variable(labels)
        optimizer.zero_grad()
        output = net(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

    print('Epoch: {}; Loss: {}'.format(epoch, loss))
