import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from vae import VAE
from data import Dataset
import os
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import argparse
import sys

def criterion(logits, aux_logits, tgt, ignore_index, mean, logvar, vocab_size):
    logits = logits.contiguous().view(-1, vocab_size)
    aux_logits = aux_logits.contiguous().view(-1, vocab_size)
    tgt = tgt.view(-1)
    cnn_loss = F.cross_entropy(aux_logits, tgt, ignore_index=vocab['<PAD>']).mean()
    rnn_loss = F.cross_entropy(logits, tgt, ignore_index=vocab['<PAD>']).mean()
    kl_loss = (-0.5*torch.sum(logvar-torch.pow(mean, 2)-torch.exp(logvar)+1, dim=1)).mean()

    return cnn_loss, rnn_loss, kl_loss
    
def id2sent(s, vocab_inv):
    return ' '.join(['' if vocab_inv[i] == '<PAD>' else vocab_inv[i] for i in s])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',       default='./data_20k/', type=str)
    parser.add_argument('--batch_size',     default=16, type=int)
    parser.add_argument('--doc_max_len',    default=180, type=int)
    parser.add_argument('--emb_size',       default=1024, type=int)
    parser.add_argument('--hid_size',       default=512, type=int)
    parser.add_argument('--latent_size',    default=512, type=int)
    parser.add_argument('--rnn_size',       default=2048, type=int)
    parser.add_argument('--rnn_num_layers', default=1, type=int)
    parser.add_argument('--lr',             default=0.0001, type=float)
    parser.add_argument('--w_cnn_loss',     default=0.1, type=float)
    parser.add_argument('--w_rnn_loss',     default=1.0, type=float)
    parser.add_argument('--cnn_type',       default='1', type=str)

    args = parser.parse_args()
    import wandb
    wandb.init(
        project='CNN VAE test',
        config = args
    )

    exit()

    config = {
        'data_dir'          :'./data_20k/',
        'batch_size'        :16,
        'doc_max_len'       :81, ## 1 -> 180, 2-> 81
        'emb_size'          :1024,
        'hid_size'          :512,
        'latent_size'       :512,
        'rnn_size'          :2048,
        'rnn_num_layers'    :1,
        'lr'                :0.0001,
        'w_cnn_loss'        :0.1,
        'w_rnn_loss'        :1,
        'w_kl_loss_rate'    :0.01,
        'cnn_type': '2',
    }
    w_kl_loss = 0
    use_wandb = True

    print('[INFO] load vocabfile')
    vocab = json.load(open(os.path.join(args.data_dir, 'vocab.json'), 'r'))
    vocab_inv = {a:b for b, a in vocab.items()}
    print('[INFO] load train data')
    train_loader = DataLoader(
        Dataset(os.path.join(args.data_dir, 'train_seq.json'),
            doc_max_len=args.doc_max_len, pad_idx=vocab['<PAD>']), 
        batch_size=args.batch_size, shuffle=True
    )
    print('[INFO] load valid data')
    valid_loader = DataLoader(
        Dataset(os.path.join(args.data_dir, 'valid_seq.json'), 
            doc_max_len=args.doc_max_len, pad_idx=vocab['<PAD>']),
        batch_size=args.batch_size, shuffle=False
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    epochs = 20
    
    model = VAE(
        vocab_size=len(vocab),
        emb_size = args.emb_size,
        hid_size = args.hid_size,
        latent_size = args.latent_size,
        padding_idx = vocab['<PAD>'],
        rnn_size = args.rnn_size,
        rnn_num_layers = args.rnn_num_layers,
        cnn_type = args.cnn_type,
        ).to(device)

    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)

    if use_wandb:
        import wandb
        wandb.init(
            project='CNN VAE test',
            config = args
        )
        
    for epoch in range(epochs):

        train_bar = tqdm(train_loader, desc="[EPOCH] {}".format(epoch))

        train_loss, total = 0, 0
        model.train()

        for batch, (document, summary) in enumerate(train_bar, 1):

            optimizer.zero_grad()

            document = document.to(device)
            # print(document.shape)
            logits, aux_logits, mean, logvar = model(document)

            logits = logits[:, :config['doc_max_len']]
            aux_logits = aux_logits[:, :config['doc_max_len']]

            cnn_loss, rnn_loss, kl_loss = criterion(logits, aux_logits, document, vocab['<PAD>'], mean, logvar, len(vocab))
            
            loss = cnn_loss*config['w_cnn_loss'] + rnn_loss*config['w_rnn_loss'] + kl_loss*w_kl_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            total += document.shape[0]
            train_bar.set_postfix(
                cnn_loss = '{:.03f}'.format(cnn_loss.item()),
                rnn_loss = '{:.03f}'.format(rnn_loss.item()),
                kl_loss = '{:.03f}'.format(kl_loss.item()),
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
                'kl loss': min(kl_loss.item(), 1e4),
                'ave loss': train_loss/total,
                'mean': mean[0][0].item(),
                'logvar': logvar[0][0].item(),
                'w_kl_loss': w_kl_loss
            }

            if use_wandb:
                wandb.log(info)
            else:
                print(info)

        continue

        valid_loss, total = 0, 0
        model.eval()
        valid_bar = tqdm(valid_loader, desc='[EVALI]')
        for batch, (document, summary) in enumerate(valid_bar, 1):

            document = document.to(device)
            logits, aux_logits, mean, logvar = model(document)

            logits = logits[:, :config['doc_max_len']]
            aux_logits = aux_logits[:, :config['doc_max_len']]

            
            cnn_loss, rnn_loss, kl_loss = criterion(logits, aux_logits, document, vocab['<PAD>'], mean, logvar, len(vocab))
            loss = cnn_loss*config['w_cnn_loss'] + rnn_loss*config['w_rnn_loss'] + kl_loss*w_kl_loss

            valid_loss += loss.item()
            total += document.shape[0]
            valid_bar.set_postfix(
                cnn_loss = '{:.03f}'.format(cnn_loss.item()),
                rnn_loss = '{:.03f}'.format(rnn_loss.item()),
                kl_loss = '{:.03f}'.format(kl_loss.item()),
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
                    'val kl loss': min(kl_loss.item(), 1e4),
                    'val ave loss': valid_loss/total,
                    'mean': mean[0].detach().cpu().numpy()[:10],
                    'logvar': logvar[0].detach().cpu().numpy()[:10],
                }

                if use_wandb:
                    wandb.log(info)
                else:
                    print(info)
        
        w_kl_loss = min(config['w_kl_loss_rate']+w_kl_loss, 1)
