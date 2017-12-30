import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


torch.manual_seed(1)
lin = nn.Linear(in_features=5, out_features=3)
data = autograd.Variable(torch.randn(2, 5))
print(lin(data))

data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
        ("Give it to me".split(), "ENGLISH"),
        ("No creo que sea una buena idea".split(), "SPANISH"),
        ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]

test_data = [("Yo creo que si".split(), "SPANISH"),
             ("it is lost on me".split(), "ENGLISH")]

word_to_ix = {}
for sent, _ in data + test_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)

label_to_ix = {'SPANISH': 0, 'ENGLISH': 1}

VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = 2


class BowClassifier(nn.Module):

    def __init__(self, num_labels, vocab_size):
        super(BowClassifier, self).__init__()
        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vec):
        return F.log_softmax(self.linear(bow_vec), dim=0)


def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sentence:
        vec[word_to_ix[word]] += 1
    return vec


def make_target(label, label_to_ix):
    return torch.LongTensor(label_to_ix[label])


model = BowClassifier(NUM_LABELS, VOCAB_SIZE)

for param in model.parameters():
    print(param)

sample = data[0]
bow_vector = make_bow_vector(sample[0], word_to_ix)
log_probs = model(autograd.Variable(bow_vector))
print(log_probs)
