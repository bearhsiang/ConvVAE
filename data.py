from torch.utils import data
import json
import numpy as np
class Dataset(data.Dataset):
    def __init__(self, filename, doc_max_len=20, tle_max_len=20, pad_idx=0):
        self.data = json.load(open(filename, 'r'))
        self.len = len(self.data['document'])
        self.doc_max_len = doc_max_len
        self.tle_max_len = tle_max_len
        self.pad_idx = pad_idx
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        doc = self.cut(self.data['document'][idx], self.doc_max_len)
        tle = self.cut(self.data['title'][idx], self.tle_max_len)
        return (np.array(doc), np.array(tle))
    def cut(self, s, l):
        if len(s) >= l:
            return s[:l]
        return s+[self.pad_idx]*(l-len(s))
