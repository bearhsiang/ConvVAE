import torch as t
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, vocab_size, latent_size, rnn_size, rnn_num_layers, embed_size):
        super().__init__()

        self.vocab_size = vocab_size
        self.latent_size = latent_size
        self.rnn_size = rnn_size
        self.embed_size = embed_size
        self.rnn_num_layers = rnn_num_layers

        self.cnn = nn.Sequential(
            nn.ConvTranspose1d(self.latent_variable_size, 512, 4, 2, 0),
            nn.BatchNorm1d(512),
            nn.ELU(),

            nn.ConvTranspose1d(512, 512, 4, 2, 0, output_padding=1),
            nn.BatchNorm1d(512),
            nn.ELU(),

            nn.ConvTranspose1d(512, 256, 4, 2, 0),
            nn.BatchNorm1d(256),
            nn.ELU(),

            nn.ConvTranspose1d(256, 256, 4, 2, 0, output_padding=1),
            nn.BatchNorm1d(256),
            nn.ELU(),

            nn.ConvTranspose1d(256, 128, 4, 2, 0),
            nn.BatchNorm1d(128),
            nn.ELU(),

            nn.ConvTranspose1d(128, self.vocab_size, 4, 2, 0)
        )

        self.rnn = nn.GRU(input_size=self.vocab_size + self.embed_size,
                          hidden_size=self.rnn_size,
                          num_layers=self.rnn_num_layers,
                          batch_first=True)

        self.hidden_to_vocab = nn.Linear(self.rnn_size, self.vocab_size)

    def forward(self, latent):
        latent = latent.unsqeeze(-1)
        ## (batch, channel = latent_size, len = 1)
        logits = self.cnn(latent)
        logits = logits.transpose(1, 2)
        return logits