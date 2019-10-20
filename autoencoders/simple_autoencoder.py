import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

batch_size = 32
learning_rate = 1e-3
num_epochs = 20

# Initial simple transformation to tensor.
# Normalization may be needed.
transforms = transforms.Compose(
    [transforms.ToTensor()]
)

dataset = torchvision.datasets.FashionMNIST(
    './data', train=True, download=True, transform=transforms)

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=2)

dataiter = iter(dataloader)
images, labels = dataiter.next()


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 128)
        self.fc4 = nn.Linear(128, 784)

    def scale(self, x):
        mins = x.min(dim=1)[0].unsqueeze(1)
        maxs = x.max(dim=1)[0].unsqueeze(1)
        return (x - mins) / (maxs - mins)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.scale(x)
        return x.view(-1, 1, 28, 28)


autoencoder = Autoencoder()
autoencoder.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 500 == 499:
            print('[Epoch: {:2d}, Batch: {:5d}] loss: {:.4f}'.format(
                epoch + 1, i + 1, running_loss / 500))
            running_loss = 0.0
