# Building an LSTM based on embeddings or one-hot-encoded sequences.
#
# We create two LSTM-based classes, the first one is based on embeddings,
# while the second one receives one-hot encoded inputs.
#
# Both models return the tag softmax for each word. This happens because of
# this sequence of operations:
#
# 1) lstm_out, self.hidden = self.lstm(...)
# 2) tag_space = self.fc(lstm_out.view(len(inputs), -1))
#
# lstm_out contains the h_t values for each time point. If we used instead
# self.hidden[0], we would be working only with the last h_t, which is pre-
# sumably more appropriate for sentiment-analysis-like problems.
#
# OPEN QUESTIONS:
# --------------
#
# 1) What is the role of the cell state? Shall we use it as an input of a FC?
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def sentence_to_indices(seq, to_ix):
    """Transform a sentence into a LongTensor of word indices.
    Parameters:
    ----------

    seq: A sequence of words.
    to_ix: A mapping between words and indices.
    """
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return Variable(tensor)


def sentence_to_ohe(sentence, vocab_len, word_to_ix):
    """Turn a sentence into a one-hot-encoded tensor.

    Parameters:
    ----------
    sentence: A sequence of words.
    vocab_len: Integer representing the vocabulary size.
    word_to_ix: Mapping between words and indices.
    """
    output = torch.zeros(len(sentence), vocab_len)
    for word_ix, word in enumerate(sentence):
        output[word_ix, word_to_ix[word]] = 1
    return Variable(output)


training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

# Create the vocabulary by assigning to each new word the current lenght
# of the vocabulary itself.
word_to_ix = {}
for sentence, tag in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {'DET': 0, 'NN': 1, 'V': 2}


class EmbeddingTagger(nn.Module):
    """ LSTM tagger based on word embeddings"""

    def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size):
        super(EmbeddingTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, target_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.zeros(1, 1, self.hidden_dim)),
                Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, inputs):
        embeds = self.word_embeddings(inputs)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(inputs), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(inputs), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


class OneHotTagger(nn.Module):
    """LSTM tagger based on one-hot-encoded sentences."""

    def __init__(self, hidden_dim, vocab_size, target_size):
        super(OneHotTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.lstm = nn.LSTM(vocab_size, hidden_dim)
        self.fc = nn.Linear(hidden_dim, target_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.zeros(1, 1, self.hidden_dim)),
                Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, inputs):
        lstm_out, self.hidden = self.lstm(
            inputs.view(len(inputs), 1, self.vocab_size), self.hidden)
        tag_space = self.fc(lstm_out.view(len(inputs), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


# Example of output of the embedding tagger
# Our input sequence is of length 5, we use an embedding dimension of 5, and a
# hidden dimension of 4.
EMBEDDING_DIM = 4
HIDDEN_DIM = 5
VOCAB_SIZE = len(word_to_ix)
TARGET_SIZE = len(tag_to_ix)

hidden = (Variable(torch.zeros(1, 1, HIDDEN_DIM)),
          Variable(torch.zeros(1, 1, HIDDEN_DIM)))

emb_tag = EmbeddingTagger(embedding_dim=5, hidden_dim=4,
                          vocab_size=len(word_to_ix),
                          target_size=len(tag_to_ix))
ohe_tag = OneHotTagger(hidden_dim=4,
                       vocab_size=len(word_to_ix),
                       target_size=len(tag_to_ix))

emb_input = sentence_to_indices(training_data[0][0], word_to_ix)
ohe_input = sentence_to_ohe(training_data[0][0], VOCAB_SIZE, word_to_ix)

emb_output = emb_tag(emb_input)
ohe_output = ohe_tag(ohe_input)
