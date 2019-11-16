import json
from tqdm import tqdm
import os
import numpy as np
from subprocess import check_output

class Preprocessor:

    def __init__(self, vocab=None, tokenizer_type='default', verbose=True, to_lower=True):
        self.to_lower = to_lower
        self.verbose = verbose
        self.vocab = vocab
        self.tokenizer_type = tokenizer_type 
        self.tokenize = self._init_tokenize()

    def _verbose(self, *args):
        if self.verbose:
            print('[INFO]', *args)

    def sentence2id(self, s, addbos=True, addeos=True):
        if self.vocab == None:
            print('VOCAB not found, please use self.make_vocab() to create vocab.')
            return
        if self.to_lower:
            s = s.lower()
        t = self.tokenize(s)
        ids = [self.vocab.get(w, self.vocab['<UNK>']) for w in t]
        if addbos:
            ids = [self.vocab['<BOS>']] + ids
        if addeos:
            ids = ids + [self.vocab['<EOS>']]
        return ids

    def make_vocab(self, data_list, N = 20000, special_list=['<BOS>', '<EOS>', '<UNK>', '<PAD>'], store_path=None):
        self._verbose('make vacab w/', *data_list, ', N =', N)
        record = {}
        for i in data_list:
            for line in tqdm(open(i, 'r'), total=getlines(i)):
                if self.to_lower:
                    line = line.lower()
                line = self.tokenize(line)
                for w in line:
                    record[w] = record.get(w, -1) + 1

        words = sorted(record, key=lambda w: record[w], reverse=True)
        words = special_list + words
        if N > 0:
            words = words[:N]
        vocab = {}
        for index, w in enumerate(words):
            vocab[w] = index
        self.vocab = vocab
        if store_path != None:
            json.dump(self.vocab, open(store_path, 'w'))
            self._verbose('dump vocab file to '+store_path)
        self._verbose('finish')

    def _init_tokenize(self):
        if self.tokenizer_type == 'default':
            return self.default_tokenize
        if self.tokenizer_type == 'nltk':
            from nltk.tokenize import WordPunctTokenizer
            return WordPunctTokenizer().tokenize
        print('tokenizer_type:'+self.tokenizer_type+'not found, use default tokenizer')
        return self.default_tokenize

    def default_tokenize(self, l):
        return l.lower().strip().split()

def getlines(name):
    return int(check_output(["wc", "-l", name]).split()[0])

if __name__ == '__main__':

    data_dir = '/hdd/giga/'
    article_file = os.path.join(data_dir, 'train.article.txt')
    title_file = os.path.join(data_dir, 'train.title.txt')

    out_dir = 'data_20k'
    os.makedirs(out_dir, exist_ok=True)

    vocab = None
    vocab_file = os.path.join(out_dir, 'vocab.json')
    # vocab = json.load(open(vocab_file, 'r'))
    P = Preprocessor(vocab=vocab, tokenizer_type='nltk')
    if vocab == None:
        vocab_data=[article_file, title_file]
        P.make_vocab(vocab_data, store_path=vocab_file)

    document = [P.sentence2id(l) for l in tqdm(open(article_file), total=getlines(article_file))]
    title = [P.sentence2id(l) for l in tqdm(open(title_file), total=getlines(title_file))]

    total = len(document)

    valid_ratio = 0.01
    valid_num = int(valid_ratio * total)
    shuffle_seed = np.arange(total)
    np.random.shuffle(shuffle_seed)

    train_set = {'document':[], 'title': []}
    valid_set = {'document':[], 'title': []}
    
    shuffled_document = []
    shuffled_title = []

    for i in tqdm(range(total)):
        index = shuffle_seed[i]
        shuffled_document.append(document[index])
        shuffled_title.append(title[index])
    
    valid_set = {
        'document': shuffled_document[:valid_num],
        'title': shuffled_title[:valid_num]
    }
    train_set = {
        'document': shuffled_document[valid_num:],
        'title': shuffled_title[valid_num:]
    }

    json.dump(train_set, open(os.path.join(out_dir, 'train_seq.json'), 'w'))
    json.dump(valid_set, open(os.path.join(out_dir, 'valid_seq.json'), 'w'))

