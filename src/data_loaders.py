import json
import logging
import sys
from collections import defaultdict
from random import choice, shuffle

import nltk.tokenize
import torch
import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import pandas as pd
from transformers import AutoTokenizer

SS = '[SS]'
SE = '[SE]'
OS = '[OS]'
OE = '[OE]'

space = lambda s: ' ' + s + ' '
logger = logging.getLogger(__file__)
def create_samples_from_data(line, tokenizer):
    return tokenizer(line['s1'])['input_ids'], tokenizer(line['s2'])['input_ids']


def get_indexes(line, indexes):
    # if 30524 not in line[0] or 30524 not in line[1]:
    #     print('here')
    return ([line[0].index(x) for x in indexes], [line[1].index(x) for x in indexes])


class few_shot_dataset(Dataset):
    def __init__(self, data, tokenizer):
        data = data
        indexes = tokenizer('{} {}'.format(SS, OS))['input_ids'][1:-1]

        unfiltered = data.apply(lambda line: create_samples_from_data(line, tokenizer), axis=1)
        mask = unfiltered.apply(lambda line: (len(line[1]) < 512 and len(line[0]) < 512) and all([x in line[0] for x in indexes]) and all([x in line[1] for x in indexes]))
        self.samples = unfiltered[mask]
        self.indexes = self.samples.apply(lambda line: get_indexes(line, indexes))
        self.labels = data['l'][mask]
    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        return self.samples.iloc[idx], self.indexes.iloc[idx], self.labels.iloc[idx]

def fix_sentence(sent, s, o):
    if s[0] > o[0]:
        sent = sent[:s[0]] + space(SS) +sent[s[0]:s[1]] + space(SE) + sent[s[1]:]
        sent = sent[:o[0]] + space(OS) +sent[o[0]:o[1]] + space(OE) + sent[o[1]:]
    else:
        sent = sent[:o[0]] + space(OS) +sent[o[0]:o[1]] + space(OE) + sent[o[1]:]
        sent = sent[:s[0]] + space(SS) + sent[s[0]:s[1]] + space(SE) + sent[s[1]:]
    return sent.strip()


def gen_positive(rels, size):
    samples = []
    for rel in rels:
        for i, r in enumerate(rels[rel]):
            if i >= size:
                break
            cur = r
            while r == cur:
                cur = choice(rels[rel])
            samples.append({'s1':r, 's2': cur, 'l':1, 't':(rel, rel)}) # cur

    return samples

def gen_negative(rels, size, ratio):
    samples = []
    all_rels = list(rels.keys())
    for rel in rels:
        for i, r in enumerate(rels[rel]):
            if i >= size:
                break
            for j in range(ratio):
                cur_rel = rel
                while rel == cur_rel:
                    cur_rel = choice(all_rels)
                cur_r = choice(rels[cur_rel])
                samples.append({'s1':r, 's2': cur_r, 'l':0, 't':(rel, cur_rel)})

    return samples


def get_focused_sentence(s):
    # for cur_s in s.split('\n'):
    if 'the special purpose transport air fleet of the moscow air' in s.lower():
        print('here')
    for cur_s in nltk.tokenize.sent_tokenize(s):
        if SS in cur_s and OS in cur_s:
            return cur_s
    return ''


def generate_samples(rels, per_rel_size, rel_data, sentences, ratio):

    model_name = 'SpanBERT/spanbert-base-cased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    new_dict = defaultdict(list)
    relevant_rels = [r for r in rels if len(rel_data[r]) >= per_rel_size]
    max_sent_leangth = 100
    max_sent_token = 450
    for rel in tqdm.tqdm(relevant_rels):
        for i, r in enumerate(rel_data[rel]):
            sent = sentences[r['docid']]
            input_sentence = fix_sentence(sent, r['s']['boundaries'], r['o']['boundaries'])
            # input_sentence = get_focused_sentence(input_sentence)
            # if not input_sentence:
            #     continue

            if len(input_sentence.split(' ')) > max_sent_leangth and len(tokenizer.encode(input_sentence)) > max_sent_token:
            # if len(input_sentence.split(' ')) > max_sent_leangth:
                input_sentence = get_focused_sentence(input_sentence)
                if not input_sentence:
                    continue
            new_dict[rel].append(input_sentence)
    for rel in new_dict:
        if len(new_dict[rel]) < 2:
            del new_dict[rel]
    pos = gen_positive(new_dict, per_rel_size)
    neg = gen_negative(new_dict, per_rel_size, ratio)
    total = pos + neg
    shuffle(total)
    return pd.DataFrame.from_records(total)
    df.to_json('train.json')

def generate_supervised_samples(rels, per_rel_size, rel_data, sentences, ratio, ):
    new_dict = defaultdict(list)
    relevant_rels = [r for r in rels if len(rel_data[r]) >= per_rel_size]
    for rel in relevant_rels:
        for i, r in enumerate(rel_data[rel]):
            sent = sentences[r['docid']]
            input_sentence = fix_sentence(sent, r['s']['boundaries'], r['o']['boundaries'])
            new_dict[rel].append(input_sentence)

    train_dict = defaultdict(list)
    test_dict = defaultdict(list)

    for rel in new_dict:
        train_dict[rel] = new_dict[rel][:-50]
        train_dict[rel] = new_dict[rel][-50:]
    pos = gen_positive(train_dict, per_rel_size)
    neg = gen_negative(train_dict, per_rel_size, ratio)
    total_train = pos + neg
    shuffle(total_train)
    pos = gen_positive(train_dict, per_rel_size)
    neg = gen_negative(train_dict, per_rel_size, ratio)
    total_test = pos + neg
    shuffle(total_test)
    return pd.DataFrame.from_records(total_train), pd.DataFrame.from_records(total_test)
    df.to_json('train.json')


def load_data(rel_path, sentence_path, rel_num=100, pos_ratio=1):
    with open(rel_path) as f:
        rels = json.load(f)
        rels = {x: y for x,y in rels.items() if y[0] != 1}
    with open(sentence_path) as f:
        sentences = json.load(f)

    train_rels = [x for x,y in rels.items() if len(y) > rel_num]
    test_rels = [x for x,y in rels.items() if len(y) < 40 and len(y) > 2]
    # train_rels = train_rels[:500]
    logger.info(f'{len(train_rels)} diffrent rels in training')
    per_rel_size = int(50000 / len(train_rels))
    train = generate_samples(train_rels, per_rel_size, rels, sentences, pos_ratio)  # [:2000]
    per_rel_size = 20
    test = generate_samples(test_rels, per_rel_size, rels, sentences, pos_ratio)  # [:2000]
    # per_rel_size = 130
    # train, test = generate_supervised_samples(train_rels, per_rel_size, rels, sentences, pos_ratio)
    return train, test, len(train_rels)
    train.to_json('train.json')
    test.to_json('train.json')
    # print(sentences)

def collate_wrapper(batch):
    features, index, label = list(zip(*batch))
    f1, f2 = list(zip(*features))
    tensor_f1 = pad_sequence([torch.tensor(x) for x in f1]).T
    tensor_f2 = pad_sequence([torch.tensor(x) for x in f2]).T
    i1, i2 = list(zip(*index))
    return tensor_f1.cuda(), tensor_f2.cuda(), i1, i2, torch.tensor([int(x) for x in label]).cuda()


def load_dataloaders(args, tokenizer):
    if 'linux' in sys.platform:
        dir = '/home/nlp/amirdnc/data/fs_rc/'
        rel_path = dir + r"rels.json"
        sentence_path = dir + r"text.json"
    else:
        dir = r'../fewshot_RC/'
        rel_path = dir + r"val_rels.json"
        sentence_path = dir + r"val_text.json"

    train_batch_size = args.batch_size
    test_batch_size = 64
    tokenizer.add_tokens([SS, SE, OS, OE])
    train, test = load_data(rel_path, sentence_path)
    # train = train[:5000]
    train_data = few_shot_dataset(train, tokenizer)
    test_data = few_shot_dataset(test, tokenizer)
    train_loader = DataLoader(train_data, batch_size=train_batch_size, collate_fn=collate_wrapper, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, collate_fn=collate_wrapper, shuffle=True)
    return train_loader, test_loader, len(train_loader)*train_batch_size, len(test_loader)*test_batch_size