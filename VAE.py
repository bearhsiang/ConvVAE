import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Encoder
from decoder import Decoder

class VAE(nn.module):

    def __init__(self, batch_size, vocab_size, emb_size, hid_size, latent_size, padding_idx):
        super.__init__()

        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.latent_size = latent_size

        self.emb_layer = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.encoder = Encoder(emb_size, hid_size)

        self.mean_creater = nn.Linear(hid_size, latent_size)
        self.var_creater = nn.Linear(hid_size, latent_size)
        
        self.dist_sample = self._dist_sample
        self.decoder = nn.Sequential()

    def forward(self, x):
        emb = self.emb_layer(x)
        hid = self.encoder(emb)
        mean = self.mean_creater(hid)
        var = self.var_createt(hid)
        latent = self.dist_sample(mean, var)
        out = self.decoder(latent)

        return out

    def _dist_sample(self, mu, std):
        z = torch.randn([self.batch_size, self.latent_size])
        z = (z + mu) / std
        return z