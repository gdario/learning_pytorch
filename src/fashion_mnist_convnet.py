import torch.nn as nn
import torch.optim as optim
import simple_convnet as sc
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

NUM_EPOCHS = 30
USE_CUDA = True


def as_cuda_variable(tensor, use_cuda=USE_CUDA):
    if torch.cuda.is_available() and use_cuda:
        tensor = tensor.cuda()
    return Variable(tensor)


def generate_dataset(mean=0, sd=16, batch_size=64,
                     data_set=datasets.FashionMNIST):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean, mean, mean, mean), (sd, sd, sd, sd))
    ])
    train_set = data_set(root='./data', train=True,
                         transform=transform, download=True)
    test_set = data_set(root='./data', train=False,
                        transform=transform, download=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    return train_set, test_set, train_loader, test_loader


def train(num_epochs=NUM_EPOCHS, num_steps=100, trainloader=None,
          net=None, criterion=None, optimizer=None):
    for epoch in range(num_epochs):
        running_loss = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = as_cuda_variable(inputs), as_cuda_variable(labels)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            if i % num_steps == num_steps - 1:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / num_steps))
                running_loss = 0.0


def compute_accuracy(data_loader=None):
    correct = 0
    total = 0
    for batch, data in enumerate(data_loader):
        images, labels = data
        images, labels = as_cuda_variable(images), as_cuda_variable(labels)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
    print('Accuracy: {}'.format(100 * correct / total))


if __name__ == '__main__':

    train_set, test_set, train_loader, test_loader = generate_dataset()
    net = sc.SimpleConvnet()
    if torch.cuda.is_available() and USE_CUDA:
        # net = nn.DataParallel(net)
        net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
    train(trainloader=train_loader, net=net, criterion=criterion,
          optimizer=optimizer)
    print('---Training Set---')
    compute_accuracy(train_loader)
    print('---Test Set---')
    compute_accuracy(test_loader)
