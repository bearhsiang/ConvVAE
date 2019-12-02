import torch
import torch.nn as nn
import torch.nn.functional as F

def encoder_cnn(cnn_type, in_channel, out_channel):
    
    cnn = None
    
    if cnn_type == '1':
        cnn = nn.Sequential(
            nn.Conv1d(in_channel, 128, 3, 2),
            # nn.BatchNorm1d(128),
            nn.ELU(),

            nn.Conv1d(128, 256, 3, 2),
            # nn.BatchNorm1d(256),
            nn.ELU(),

            nn.Conv1d(256, 256, 3, 2),
            # nn.BatchNorm1d(256),
            nn.ELU(),

            nn.Conv1d(256, 512, 3, 2),
            # nn.BatchNorm1d(512),
            nn.ELU(),

            nn.Conv1d(512, 512, 3, 2),
            # nn.BatchNorm1d(512),
            nn.ELU(),

            nn.Conv1d(512, out_channel, 3, 2),
            # nn.BatchNorm1d(out_channel),
            nn.ELU()
        )
    elif cnn_type == '2':
        ## len 81 -> 1
        cnn = nn.Sequential(
            nn.Conv1d(in_channel, 128, 3, 3),
            nn.BatchNorm1d(128),
            nn.ELU(),
            
            nn.Conv1d(128, 256, 3, 3),
            nn.BatchNorm1d(256),
            nn.ELU(),
            
            nn.Conv1d(256, 512, 3, 3),
            nn.BatchNorm1d(512),
            nn.ELU(),
            
            nn.Conv1d(512, out_channel, 3, 3),
            nn.BatchNorm1d(out_channel),
            nn.ELU(),
        )
        
    return cnn

def decoder_cnn(cnn_type, in_channel, out_channel):
    
    cnn = None
    
    if cnn_type == '1':
        cnn = nn.Sequential(
            ## (in_channel , out_channel, k_size, stride, padding, out padding)
            nn.ConvTranspose1d(in_channel, 512, 4, 2, 0),
            # nn.BatchNorm1d(512),
            nn.ELU(),
            ## len = 4
            nn.ConvTranspose1d(512, 512, 4, 2, 0, output_padding=1),
            # nn.BatchNorm1d(512),
            nn.ELU(),
            ## len = 11
            nn.ConvTranspose1d(512, 256, 4, 2, 0),
            # nn.BatchNorm1d(256),
            nn.ELU(),
            ## len = 24
            nn.ConvTranspose1d(256, 256, 4, 2, 0, output_padding=1),
            # nn.BatchNorm1d(256),
            nn.ELU(),
            ## len = 51
            nn.ConvTranspose1d(256, 128, 4, 2, 0),
            # nn.BatchNorm1d(128),
            nn.ELU(),
            ## len = 104
            nn.ConvTranspose1d(128, out_channel, 4, 2, 0)
            ## len = 207
        )
    elif cnn_type == '2':
        ## len 1 => 81
        cnn = nn.Sequential(
            
            nn.ConvTranspose1d(in_channel, 512, 3, 3),
            nn.BatchNorm1d(512),
            nn.ELU(),
            
            nn.ConvTranspose1d(512, 256, 3, 3),
            nn.BatchNorm1d(256),
            nn.ELU(),
            
            nn.ConvTranspose1d(256, 128, 3, 3),
            nn.BatchNorm1d(128),
            nn.ELU(),
            
            nn.ConvTranspose1d(128, out_channel, 3, 3),
        )
        
    return cnn