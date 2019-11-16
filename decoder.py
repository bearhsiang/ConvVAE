import torch as t
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, vocab_size, latent_size, rnn_size, rnn_num_layers):
        super().__init__()

        self.vocab_size = vocab_size
        self.latent_size = latent_size
        self.rnn_size = rnn_size
        self.rnn_num_layers = rnn_num_layers

        self.cnn = nn.Sequential(
            ## (in_channel , out_channel, k_size, stride, padding, out padding)
            nn.ConvTranspose1d(self.latent_variable_size, 512, 4, 2, 0),
            nn.BatchNorm1d(512),
            nn.ELU(),
            ## len = 4
            nn.ConvTranspose1d(512, 512, 4, 2, 0, output_padding=1),
            nn.BatchNorm1d(512),
            nn.ELU(),
            ## len = 11
            nn.ConvTranspose1d(512, 256, 4, 2, 0),
            nn.BatchNorm1d(256),
            nn.ELU(),
            ## len = 24
            nn.ConvTranspose1d(256, 256, 4, 2, 0, output_padding=1),
            nn.BatchNorm1d(256),
            nn.ELU(),
            ## len = 51
            nn.ConvTranspose1d(256, 128, 4, 2, 0),
            nn.BatchNorm1d(128),
            nn.ELU(),
            ## len = 104
            nn.ConvTranspose1d(128, self.vocab_size, 4, 2, 0)
            ## len = 207
        )

        self.rnn = nn.GRU(input_size=self.vocab_size,
                          hidden_size=self.rnn_size,
                          num_layers=self.rnn_num_layers,
                          batch_first=True)

        self.rnn_to_vocab = nn.Linear(self.rnn_size, self.vocab_size)

    def forward(self, latent):
        aux_logits = self._decode_cnn(latent)
        logits = self._decode_rnn(aux_logits)
        return logits, aux_logits
    
    def _decode_cnn(self, latent):        
        latent = latent.unsqeeze(-1)
        ## (batch, channel = latent_size, len = 1)
        logits = self.cnn(latent)
        logits = logits.transpose(1, 2)
        ## (batch, len, vocab)
        return logits

    def _decode_rnn(self, cnn_out):
        rnn_out, _ = self.rnn(cnn_out)
        logits = self.rnn_to_vacab(rnn_out)
        ## (batch, len, vacab)
        return logits
