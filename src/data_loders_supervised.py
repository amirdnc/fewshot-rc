import json
from random import shuffle

from torch.utils.data import Dataset

from data_loaders import fix_sentence, create_samples_from_data, get_indexes
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
import pandas as pd
SS = '[SS]'
SE = '[SE]'
OS = '[OS]'
OE = '[OE]'

def prosses_rel(rel, sentences):
    sent = sentences[rel['docid']]
    return fix_sentence(sent, rel['s']['boundaries'], rel['o']['boundaries'])


def load_supervised_data(rel_path, sentence_path):
    with open(rel_path) as f:
        rels = json.load(f)
        rels = {x: y for x,y in rels.items() if y[0] != 1}
    with open(sentence_path) as f:
        sentences = json.load(f)

    train_rels = [x for x,y in rels.items() if len(y) > 15000]
    labels = {x: i for i,x in enumerate(train_rels)}
    train = []
    for rel_type in train_rels:
        for rel in rels[rel_type][:1000]:
            train.append({'sentence': prosses_rel(rel, sentences), 'label': labels[rel_type]})
    shuffle(train)
    return train, labels


class supervised_dataset(Dataset):
    def __init__(self, data, tokenizer):
        data = pd.DataFrame.from_records(data)
        tokenizer.add_tokens([SS, SE, OS, OE])
        indexes = tokenizer('{} {}'.format(SS, OS))['input_ids'][1:-1]

        unfiltered = data.apply(lambda line: tokenizer(line['sentence'])['input_ids'], axis=1)
        mask = unfiltered.apply(lambda line: (len(line) < 512) and all([x in line for x in indexes]))
        self.samples = unfiltered[mask]
        self.indexes = self.samples.apply(lambda line: [line.index(x) for x in indexes])
        self.labels = data['label'][mask]
    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        return self.samples.iloc[idx], self.indexes.iloc[idx], self.labels.iloc[idx]

def collate_wrapper(batch):
    features, index, label = list(zip(*batch))
    tensor_f1 = pad_sequence([torch.tensor(x) for x in features])
    return tensor_f1.cuda(), index, torch.tensor([int(x) for x in label]).cuda()