import torch as t
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, embed_size, latent_size):
        super(Encoder, self).__init__()

        self.embed_size = embed_size
        self.latent_size = latent_size

        self.cnn = nn.Sequential(
            nn.Conv1d(self.embed_size, 128, 3, 2),
            nn.BatchNorm1d(128),
            nn.ELU(),

            nn.Conv1d(128, 256, 3, 2),
            nn.BatchNorm1d(256),
            nn.ELU(),

            nn.Conv1d(256, 256, 3, 2),
            nn.BatchNorm1d(256),
            nn.ELU(),

            nn.Conv1d(256, 512, 3, 2),
            nn.BatchNorm1d(512),
            nn.ELU(),

            nn.Conv1d(512, 512, 3, 2),
            nn.BatchNorm1d(512),
            nn.ELU(),

            nn.Conv1d(512, self.latent_size, 3, 2),
            nn.BatchNorm1d(self.latent_size),
            nn.ELU()
        )

    def forward(self, input):
        
        input = torch.transpose(input, 1, 2)
        result = self.cnn(input)
        return result.squeeze(2)