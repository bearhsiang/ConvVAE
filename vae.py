import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Encoder
from decoder import Decoder

class VAE(nn.Module):

    def __init__(self, vocab_size, emb_size, hid_size, latent_size, padding_idx, rnn_size, rnn_num_layers, cnn_type):
        super().__init__()

        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.latent_size = latent_size
        self.cnn_type = cnn_type

        self.emb_layer = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.encoder = Encoder(emb_size, hid_size, cnn_type)

        self.mean_creater = nn.Linear(hid_size, latent_size)
        self.logvar_creater = nn.Linear(hid_size, latent_size)
        
        self.dist_sample = self._dist_sample
        self.decoder = Decoder(vocab_size, latent_size, rnn_size, rnn_num_layers, cnn_type)

    def forward(self, x):
        emb = self.emb_layer(x)
        hid = self.encoder(emb)
        mean = self.mean_creater(hid)
        logvar = self.logvar_creater(hid)
        latent = self.dist_sample(mean, logvar)
        logits, aux_logits= self.decoder(latent)

        return logits, aux_logits, mean, logvar

    def _dist_sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        z = torch.randn(mu.shape).type_as(mu)
        z = (z + mu) / std
        return z

class AutoEncoder(VAE):
    def forward(self, x):
        emb = self.emb_layer(x)
        hid = self.encoder(emb)
        logits, aux_logits = self.decoder(hid)
        
        return logits, aux_logits