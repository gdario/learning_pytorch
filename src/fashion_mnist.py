import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torch.nn.functional as F
# import numpy as np
import torchvision.transforms as transforms
import torch.optim as optim
# import ipdb


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

# Download the Fashion MNIST dataset
trainset = torchvision.datasets.FashionMNIST(
    root='../data/', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.FashionMNIST(
    root='../data/', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=500,
                                         shuffle=True, num_workers=2)


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.fc1 = nn.Linear(in_features=32 * 4 * 4, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = MyNet().cuda()
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=net.parameters(), lr=1e-03)

for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
        if (i + 1) % 100 == 0:
            print('Epoch: {}; Step {}; Loss: {:.3f}'.format(
                (epoch + 1), 64 * (i + 1), running_loss / (i + 1)))


# Performance on the test set
# Test on the test set
correct = 0.0
total = 0.0

net = net.cpu()

for data in testloader:
    # ipdb.set_trace()
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += 500
    correct += (predicted == labels).sum()

print('Accuracy: {}'.format(100 * correct / total))

torch.save(net.state_dict(), 'fashion_mnist.pkl')
