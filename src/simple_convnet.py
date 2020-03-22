import torch.nn as nn
import torch.nn.functional as functional


class SimpleConvnet(nn.Module):

    def __init__(self):
        super(SimpleConvnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = functional.relu(self.conv1(x))
        x = functional.max_pool2d(x, (2, 2))
        x = functional.relu(self.conv2(x))
        x = functional.max_pool2d(x, (2, 2))
        x = x.view(x.size(0), -1)
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
