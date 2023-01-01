import json
import os

import torch
from torch import nn


class TokenLocation(nn.Module):

    def __init__(self, tokens):
        super(TokenLocation, self).__init__()
        self.toekens = tokens

    def forward(self, features):
        features['indexes'] = [[(i == token).nonzero(as_tuple=True)[0] for i in features['input_ids']] for token in self.toekens]
        return features

    def save(self, output_path):
        pass

    @staticmethod
    def load(path):
        return TokenLocation([28996, 28998])  # the saved token we use. might change based on the tokenizer

class Selector(nn.Module):

    def __init__(self):
        super(Selector, self).__init__()

    def forward(self, features):
        # print((features['indexes'][1]))


        try:
            s_tokens = features['token_embeddings'][torch.arange(features['token_embeddings'].size(0)), features['indexes'][0]]
            o_tokens = features['token_embeddings'][torch.arange(features['token_embeddings'].size(0)), features['indexes'][1]]
        except:
            print('error when extracting tokens')
            device = features['token_embeddings'].device
            res = torch.zeros(features['token_embeddings'].size(0), 768 * 2).to(device)
            res.requires_grad_()
            features.update({'sentence_embedding': res})
            return features
        features.update({'sentence_embedding': torch.cat([s_tokens, o_tokens], axis=1)})
        return features

    def save(self, output_path):
        pass

    @staticmethod
    def load(path):
        return Selector()