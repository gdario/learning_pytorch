import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


torch.manual_seed(1)

word_to_ix = {'hello': 0, 'world': 1}

# vocabulary size=2, embedding dimension = 5
embeds = nn.Embedding(2, 5)
lookup_tensor = torch.LongTensor([word_to_ix['hello']])
hello_embed = embeds(autograd.Variable(lookup_tensor))
print(hello_embed)

# N-Gram Language Modeling
CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()


# Exercise: build a sentence tokenizer that can take care of punctuation.
trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]
print(trigrams[:3])

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, 1)
        return log_probs


losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(70):
    total_loss = torch.Tensor([0])
    for context, target in trigrams:
        context_idxs = [word_to_ix[w] for w in context]
        context_var = autograd.Variable(torch.LongTensor(context_idxs))

        model.zero_grad()
        log_probs = model(context_var)

        target_var = autograd.Variable(torch.LongTensor([word_to_ix[target]]))
        loss = loss_function(log_probs, target_var)
        loss.backward()
        optimizer.step()
        total_loss += loss.data
    losses.append(total_loss)
print(losses)

ix_to_word = {ix: word for (word, ix) in word_to_ix.items()}

# Let's check how the generated text looks like
for context, _ in trigrams:
    context_idxs = [word_to_ix[w] for w in context]
    context_var = autograd.Variable(torch.LongTensor(context_idxs))
    log_probs = model(context_var)
    max_prob = int(torch.max(log_probs, dim=1)[1])
    predicted_word = ix_to_word[max_prob]
    print(predicted_word, end=' ')

# plt.plot(range(50), losses, '-')
# plt.show()
