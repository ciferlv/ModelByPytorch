import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ERClassifier(nn.Module):
    def __init__(self, character_size, embed_dim, hidden_size, dense1_size, dense2_size):
        self.embedding = nn.Embedding(character_size, embed_dim)
        self.gru = nn.GRU(hidden_size, hidden_size)

        self.drop_out = nn.Dropout()
        self.dense_layer1 = nn.Linear(hidden_size, dense1_size)
        self.dense_layer2 = nn.Linear(dense1_size, dense2_size)

        self.out = nn.Linear(dense2_size, 2)

    def forward(self, input_tensor, hidden_tensor):
        output_tensor, hidden_tensor = self.gru(input_tensor, hidden_tensor)

        z1 = nn.ReLU(self.dense_layer1(output_tensor))
        z1 = self.drop_out(z1)

        z2 = nn.Relu(self.dense_layer2(z1))
        z2 = self.drop_out(z2)

        out_prob = nn.LogSoftmax(self.out(z2))
        return z2, out_prob
