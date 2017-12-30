import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


CONTEXT_SIZE = 2

raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {w: i for i, w in enumerate(vocab)}

# Creation of the context-target tuples
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))

print(data[0])


class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(2 * context_size, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        sum_embeds = torch.sum(embeds, dim=1).view(1, -1)
        fc1 = self.linear1(sum_embeds)
        # import ipdb; ipdb.set_trace()
        return F.log_softmax(fc1, dim=1)


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    tensor = torch.LongTensor(idxs)
    return Variable(tensor)


cbow = CBOW(vocab_size, 10, 2)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(cbow.parameters(), lr=0.001)
losses = []

for epoch in range(10):
    total_loss = torch.Tensor([0])
    for context, target in data:
        context_vector = make_context_vector(context, word_to_ix)
        target_vector = make_context_vector([target], word_to_ix)
        cbow.zero_grad()
        output = cbow(context_vector)
        loss = loss_function(output, target_vector)
        loss.backward()
        optimizer.step()
        total_loss += loss.data
    losses.append(total_loss)
print(losses)
