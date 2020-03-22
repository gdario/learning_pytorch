import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


cuda_is_available = True
batch_size = 32
n_epochs = 50
learning_rate = 0.001  # Default 0.001

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(
    root='data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2)
n_train_examples = len(trainset)

testset = torchvision.datasets.CIFAR10(
    root='data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2)
n_test_examples = len(testset)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.fc1 = nn.Linear(16 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 6 * 6)
        x = F.relu(F.dropout(self.fc1(x)))
        x = F.relu(F.dropout(self.fc2(x)))
        x = F.dropout(self.fc3(x))
        return x


net = Net()
if cuda_is_available:
    net = net.cuda()


def evaluate_on_dataloader(
        data_loader, optimizer, model, loss_fn, n_examples, train=True):
    """Calculate the loss of the model on a given data loader"""

    running_loss = 0.0

    for i, data in enumerate(data_loader, 0):
        inputs, labels = data
        if cuda_is_available:
            inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        if train:
            loss.backward()
            optimizer.step()
        running_loss += loss.data[0]

    return running_loss / n_examples


def compute_accuracy(data_loader, model, n_examples):
    correct = 0
    total = 0
    for i, data in enumerate(data_loader, 0):
        inputs, labels = data
        if cuda_is_available:
            inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, dim=1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
    return 100 * correct / total


# CrossEntropyLoss combines nn.LogSoftmax and nn.NLLLoss
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

for epoch in range(n_epochs):

    train_loss = evaluate_on_dataloader(
        trainloader, optimizer, net, loss_function, n_train_examples)

    test_loss = evaluate_on_dataloader(
        testloader, optimizer, net, loss_function, n_test_examples,
        train=False)
    print('Epoch: {:2d}, training loss: {:f}, test loss {:f}'.format(
        epoch, train_loss, test_loss))

train_accuracy = compute_accuracy(trainloader, net, n_train_examples)
test_accuracy = compute_accuracy(testloader, net, n_test_examples)

print('Final train accuracy: {:.2f}% - Final test accuracy: {:.2f}%'.format(
    train_accuracy, test_accuracy))

print('Saving the model')
torch.save(net.state_dict(), 'data/cifar10_tutorial.pkl')
