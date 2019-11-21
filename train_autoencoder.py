import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from vae import AutoEncoder
from data import Dataset
import os
from torch.utils.data import DataLoader
import json
from tqdm import tqdm

def criterion(logits, aux_logits, tgt, ignore_index, vocab_size):
    logits = logits.contiguous().view(-1, vocab_size)
    aux_logits = aux_logits.contiguous().view(-1, vocab_size)
    tgt = tgt.view(-1)
    cnn_loss = F.cross_entropy(aux_logits, tgt, ignore_index=vocab['<PAD>']).mean()
    rnn_loss = F.cross_entropy(logits, tgt, ignore_index=vocab['<PAD>']).mean()

    return cnn_loss, rnn_loss
    
def id2sent(s, vocab_inv):
    return ' '.join([vocab_inv[i] for i in s])

if __name__ == '__main__':

    config = {
        'data_dir'          :'./data_20k/',
        'batch_size'        :16,
        'doc_max_len'       :180, ## 1 -> 180, 2-> 81
        'emb_size'          :1024,
        'hid_size'          :512,
        'latent_size'       :512,
        'rnn_size'          :512,
        'rnn_num_layers'    :1,
        'lr'                :0.001,
        'w_cnn_loss'        :0.2,
        'w_rnn_loss'        :1,
        'w_kl_loss_rate'    :0.01,
        'cnn_type': '1',
    }
    use_wandb = True

    print('[INFO] load vocabfile')
    vocab = json.load(open(os.path.join(config['data_dir'], 'vocab.json'), 'r'))
    vocab_inv = {a:b for b, a in vocab.items()}
    print('[INFO] load train data')
    train_loader = DataLoader(
        Dataset(os.path.join(config['data_dir'], 'train_seq.json'),
            doc_max_len=config['doc_max_len'], pad_idx=vocab['<PAD>']), 
        batch_size=config['batch_size'], shuffle=True
    )
    print('[INFO] load valid data')
    valid_loader = DataLoader(
        Dataset(os.path.join(config['data_dir'], 'valid_seq.json'), 
            doc_max_len=config['doc_max_len'], pad_idx=vocab['<PAD>']),
        batch_size=config['batch_size'], shuffle=False
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    epochs = 10
    
    model = AutoEncoder(
        vocab_size=len(vocab),
        emb_size = config['emb_size'],
        hid_size = config['hid_size'],
        latent_size = config['latent_size'],
        padding_idx = vocab['<PAD>'],
        rnn_size = config['rnn_size'],
        rnn_num_layers = config['rnn_num_layers'],
        cnn_type = config['cnn_type'],
        ).to(device)

    optimizer = optim.RMSprop(model.parameters(), lr=config['lr'])

    if use_wandb:
        import wandb
        wandb.init(
            project='CNN AutoEncoder',
            config = config,
        )
        
    for epoch in range(epochs):

        train_bar = tqdm(train_loader, desc="[EPOCH] {}".format(epoch))

        train_loss, total = 0, 0
        model.train()

        for batch, (document, summary) in enumerate(train_bar, 1):

            optimizer.zero_grad()

            document = document.to(device)
            logits, aux_logits = model(document)

            logits = logits[:, :config['doc_max_len']]
            aux_logits = aux_logits[:, :config['doc_max_len']]

            cnn_loss, rnn_loss  = criterion(logits, aux_logits, document, vocab['<PAD>'], len(vocab))
            
            loss = cnn_loss*config['w_cnn_loss'] + rnn_loss*config['w_rnn_loss']

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            total += document.shape[0]
            train_bar.set_postfix(
                cnn_loss = '{:.03f}'.format(cnn_loss.item()),
                rnn_loss = '{:.03f}'.format(rnn_loss.item()),
                loss = '{:.03f}'.format(train_loss/total)
            )

            cnn_ys = aux_logits.argmax(dim=-1)
            rnn_ys = logits.argmax(dim=-1)

            info = {
                'input': id2sent(document[0].cpu().numpy(), vocab_inv),
                'cnn output': id2sent(cnn_ys[0].cpu().numpy(), vocab_inv),
                'rnn output': id2sent(rnn_ys[0].cpu().numpy(), vocab_inv),
                'cnn loss': cnn_loss.item(),
                'rnn loss': rnn_loss.item(),
                'ave loss': train_loss/total,
            }

            if use_wandb:
                wandb.log(info)
            else:
                print(info)



        valid_loss, total = 0, 0
        model.eval()
        valid_bar = tqdm(valid_loader, desc='[EVALI]')
        for batch, (document, summary) in enumerate(valid_bar, 1):

            document = document.to(device)
            logits, aux_logits = model(document)

            logits = logits[:, :config['doc_max_len']]
            aux_logits = aux_logits[:, :config['doc_max_len']]

            
            cnn_loss, rnn_loss = criterion(logits, aux_logits, document, vocab['<PAD>'], len(vocab))
            loss = cnn_loss*config['w_cnn_loss'] + rnn_loss*config['w_rnn_loss']

            valid_loss += loss.item()
            total += document.shape[0]
            valid_bar.set_postfix(
                cnn_loss = '{:.03f}'.format(cnn_loss.item()),
                rnn_loss = '{:.03f}'.format(rnn_loss.item()),
                loss = '{:.03f}'.format(valid_loss/total)
            )
            if batch == len(valid_bar):
                cnn_ys = aux_logits.argmax(dim=-1)
                rnn_ys = logits.argmax(dim=-1)
                info = {
                    'val input': id2sent(document[0].cpu().numpy(), vocab_inv),
                    'val cnn output': id2sent(cnn_ys[0].cpu().numpy(), vocab_inv),
                    'val rnn output': id2sent(rnn_ys[0].cpu().numpy(), vocab_inv),
                    'val cnn loss': cnn_loss.item(),
                    'val rnn loss': rnn_loss.item(),
                    'val ave loss': valid_loss/total,
                }

                if use_wandb:
                    wandb.log(info)
                else:
                    print(info)
