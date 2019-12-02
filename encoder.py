import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_structure import encoder_cnn

class Encoder(nn.Module):
    def __init__(self, embed_size, latent_size, cnn_type):
        super().__init__()

        self.embed_size = embed_size
        self.latent_size = latent_size

        self.cnn = encoder_cnn(cnn_type, embed_size, latent_size)

    def forward(self, input):
        
        input = torch.transpose(input, 1, 2)
        result = self.cnn(input)
        return result.squeeze(2)
