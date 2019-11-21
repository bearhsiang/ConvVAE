import torch as t
import torch.nn as nn
import torch.nn.functional as F
from cnn_structure import decoder_cnn

class Decoder(nn.Module):
    def __init__(self, vocab_size, latent_size, rnn_size, rnn_num_layers, cnn_type):
        super().__init__()

        self.vocab_size = vocab_size
        self.latent_size = latent_size
        self.rnn_size = rnn_size
        self.rnn_num_layers = rnn_num_layers

        self.cnn = decoder_cnn(cnn_type, latent_size, vocab_size)

        self.rnn = nn.GRU(input_size=self.vocab_size,
                          hidden_size=self.rnn_size,
                          num_layers=self.rnn_num_layers,
                          batch_first=True)

        self.rnn_to_vocab = nn.Linear(self.rnn_size, self.vocab_size)

    def forward(self, latent):
        aux_logits = self._decode_cnn(latent)
        # print('shape after decoder = {}'.format(aux_logits.shape))
        logits = self._decode_rnn(aux_logits)
        return logits, aux_logits
    
    def _decode_cnn(self, latent):        
        latent = latent.unsqueeze(-1)
        ## (batch, channel = latent_size, len = 1)
        logits = self.cnn(latent)
        logits = logits.transpose(1, 2)
        ## (batch, len, vocab)
        return logits

    def _decode_rnn(self, cnn_out):
        rnn_out, _ = self.rnn(cnn_out)
        logits = self.rnn_to_vocab(rnn_out)
        ## (batch, len, vacab)
        return logits
