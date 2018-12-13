# from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('./data/fra-eng/%s-%s.txt' % (lang1, lang2), encoding='utf-8'). \
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
           len(p[1].split(' ')) < MAX_LENGTH and \
           p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


# Above is to prepare train data.

#encoder
class Encoder(nn.Module):
    def __init__(self, word_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = torch.nn.Embedding(word_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size)

    def forward(self, input_tensor, hidden_tensor):
        embed_tensor = self.embedding(input_tensor).view(1, 1, -1)
        output_tensor, hidden_tensor = self.gru(embed_tensor, hidden_tensor)
        return output_tensor, hidden_tensor

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

#decoder without attention
class Decoder(nn.Module):
    def __init__(self, word_size, hidden_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = torch.nn.Embedding(word_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size)
        self.out = torch.nn.Linear(hidden_size, word_size)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, input_tensor, hidden_tensor):
        embed_tensor = self.embedding(input_tensor).view(1, 1, -1)
        embed_tensor = F.relu(embed_tensor)
        output_tensor, hidden_tensor = self.gru(embed_tensor, hidden_tensor)
        out_prob = self.softmax(self.out(output_tensor[0]))
        return hidden_tensor, out_prob

    def init_hidden_state(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def sentence2index(lang, sentence):
    idx_list = [lang.word2index[word] for word in sentence.split()]
    idx_list.append(EOS_token)
    return idx_list

# Decoder without attention
class AttenDecoder(nn.Module):
    def __init__(self, word_size, hidden_size, drop_p, max_length):
        super(AttenDecoder, self).__init__()
        self.embedding = torch.nn.Embedding(word_size, hidden_size)
        self.gru = torch.nn.GRU(input_size=hidden_size, hidden_size=hidden_size)
        self.dropout = torch.nn.Dropout(p=drop_p)
        self.attn_weight = torch.nn.Linear(hidden_size * 2, max_length)
        self.attn_combine = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.out = torch.nn.Linear(hidden_size, word_size)

    def forward(self, encoder_outputs, hidden, decoder_input):
        embed = self.embedding(decoder_input)
        embed = self.dropout(embed)

        attn = torch.cat((embed, hidden.squeeze(0)), 1)
        weights = F.softmax(self.attn_weight(attn), 1)
        attn_applied = torch.matmul(weights, encoder_outputs)

        attn_combined = torch.cat((attn_applied, embed), 1)
        gru_input = self.attn_combine(attn_combined).unsqueeze(0)
        gru_input = F.relu(gru_input)
        gru_output, gru_hidden = self.gru(gru_input, hidden)

        out_prob = F.log_softmax(self.out(gru_hidden[0]), dim=1)
        return gru_hidden, out_prob


def train_attention(encoder, decoder, encoder_optim, decoder_optim, pair, input_lang, output_lang, max_length,
                    hidden_size):
    criterion = torch.nn.NLLLoss()

    encoder.zero_grad()
    decoder.zero_grad()
    loss = 0
    source_idx_list = sentence2index(input_lang, pair[0])
    target_idx_list = sentence2index(output_lang, pair[1])

    hidden = encoder.init_hidden()
    encoder_outputs = torch.zeros(max_length, hidden_size, device=device)

    for cnt, word_idx in enumerate(source_idx_list):
        encoder_input = torch.LongTensor([word_idx], device=device)
        output, hidden = encoder.forward(encoder_input, hidden_tensor=hidden)
        encoder_outputs[cnt] = output

    decoder_input = torch.LongTensor([SOS_token], device=device)
    for word_idx in target_idx_list:
        hidden, out_prob = decoder.forward(encoder_outputs, hidden, decoder_input)
        loss += criterion(out_prob, torch.LongTensor([word_idx], device=device))
        decoder_input = torch.LongTensor([word_idx], device=device)

    loss.backward()
    encoder_optim.step()
    decoder_optim.step()

    return loss.item() / len(target_idx_list)

# complete train process to train Encoder and Decoder with Attention
def trainIter():
    epoch = 200
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    max_length = 10
    hidden_size = 50
    encoder = Encoder(input_lang.n_words, hidden_size)
    decoder = AttenDecoder(output_lang.n_words, hidden_size, 0.2, 10)
    encoder_optim = optim.Adam(encoder.parameters())
    decoder_optim = optim.Adam(decoder.parameters())

    for epoch_i in range(epoch):
        loss = 0
        cnt = 0
        for idx, pair in enumerate(pairs):
            loss += train_attention(encoder, decoder, encoder_optim, decoder_optim, pair, input_lang, output_lang,
                                    max_length,
                                    hidden_size)
            cnt += 1
            if ((idx % 1000) == 0 and idx != 0) or idx == len(pairs) - 1:
                print("Epoch: {}, Loss: {}.".format(epoch_i, loss / cnt))
                loss = 0
                cnt = 0


def train_without_attention():
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)

    encoder = Encoder(input_lang.n_words, 50)
    decoder = Decoder(output_lang.n_words, 50)
    encoder_optimizer = optim.Adam(encoder.parameters())
    decoder_optimizer = optim.Adam(decoder.parameters())

    criterion = torch.nn.NLLLoss()
    for one_pair in pairs:
        encoder.zero_grad()
        decoder.zero_grad()
        loss = 0
        input_idx = sentence2index(input_lang, one_pair[0])
        output_idx = sentence2index(output_lang, one_pair[1])
        input_hidden_tensor = encoder.init_hidden()
        for idx in input_idx:
            idx_tensor = torch.LongTensor([idx], device=device)
            output_tensor, input_hidden_tensor = encoder.forward(idx_tensor, hidden_tensor=input_hidden_tensor)

        idx_tensor = torch.LongTensor([SOS_token], device=device)
        for idx in output_idx:
            input_hidden_tensor, out_prob = decoder.forward(idx_tensor, input_hidden_tensor)
            loss += criterion(out_prob, torch.LongTensor([idx], device=device))
            idx_tensor = torch.LongTensor([idx], device=device)

        loss.backward()
        print("Loss: {}.".format(loss.item() / len(output_idx)))
        encoder_optimizer.step()
        decoder_optimizer.step()


if __name__ == "__main__":
    #Attention
    trainIter()

    #With out Attention
    # train_without_attention()
