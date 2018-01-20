# import ipdb
import numpy as np
import time
import string
import glob
import math
import random
import unicodedata
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim


def find_files(path):
    """Return the full path of the files stored in `path`."""
    return glob.glob(path)


def unicode_to_ascii(s):
    """Convert a unicode string to its ASCII equivalent."""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def read_lines(filename):
    """Read and clean the lines of `filename`."""
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]


def random_choice(lst):
    """Return a random item from a list."""
    return lst[random.randint(0, len(lst) - 1)]


def random_training_pair():
    """Return a random pair of category and category lines."""
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    return category, line


def category_tensor(category):
    """Map a category to a Torch tensor."""
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor


def input_tensor(line):
    """Map a line (i.e. a name) to a Torch tensor."""
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor


def target_tensor(line):
    """Map a target to a Torch tensor."""
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1)  # EOS
    return torch.LongTensor(letter_indexes)


def random_training_example():
    """Return a random training example as a tuple of Torch tensors."""
    category, line = random_training_pair()
    categ_tensor = Variable(category_tensor(category))
    input_line_tensor = Variable(input_tensor(line))
    target_line_tensor = Variable(target_tensor(line))
    return categ_tensor, input_line_tensor, target_line_tensor


def time_since(since):
    """Calculate the time elapsed."""
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def sample(category, start_letter='A'):
    """Sample a category and provide an initial letter for name generation."""
    categ_tensor = Variable(category_tensor(category))
    inputs = Variable(input_tensor(start_letter))
    hidden = rnn.init_hidden()

    output_name = start_letter

    for i in range(max_length):
        output, hidden = rnn(categ_tensor, inputs[0], hidden)
        probs = torch.exp(output).view(-1).data.numpy()
        idx = np.random.choice(range(n_letters), p=probs)
        if idx == n_letters - 1:
            break
        else:
            letter = all_letters[idx]
            output_name += letter
        inputs = Variable(input_tensor(letter))

    return output_name


def samples(category, start_letters='ABC'):
    """Generate one name per initial letter given the category."""
    for start_letter in start_letters:
        print(sample(category, start_letter))


n_iters = 100000
print_every = 5000
plot_every = 500
all_losses = []
total_loss = 0
learning_rate = 0.0005

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1  # Include the EOS marker

category_lines = {}
all_categories = []

for filename in find_files('../data/names/*.txt'):
    category = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    lines = read_lines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(n_categories + input_size + hidden_size,
                             hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size,
                             output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, inputs, hidden):
        input_combined = torch.cat((category, inputs, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


rnn = RNN(n_letters, 128, n_letters)

criterion = nn.NLLLoss()
optimizer = optim.Adam(params=rnn.parameters(), lr=learning_rate)
# optimizer = optim.SGD(rnn.parameters(), lr=learning_rate)


def train(category_tensor, input_line_tensor, target_line_tensor):
    """Train the model."""

    hidden = rnn.init_hidden()
    rnn.zero_grad()
    loss = 0

    for i in range(input_line_tensor.size()[0]):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        loss += criterion(output, target_line_tensor[i])

    loss.backward()
    optimizer.step()

    return output, loss.data[0] / input_line_tensor.size()[0]


start = time.time()

for iter in range(1, n_iters + 1):
    output, loss = train(*random_training_example())
    total_loss += loss

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (
            time_since(start), iter, iter / n_iters * 100, loss))

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0

plt.figure()
plt.plot(all_losses)
plt.show()

max_length = 20


samples('Italian', 'AAA')
# samples('Russian', 'RUS')
# samples('German', 'GER')
# samples('Spanish', 'SPA')
# samples('Chinese', 'CHI')
