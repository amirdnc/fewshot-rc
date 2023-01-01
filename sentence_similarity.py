"""
This examples trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) for the STSbenchmark from scratch. It generates sentence embeddings
that can be compared using cosine-similarity to measure the similarity.

Usage:
python training_nli.py

OR
python training_nli.py pretrained_transformer_model_name
"""


import os

import pandas as pd


# os.environ['CUDA_VISIBLE_DEVICES'] ='3'
rel_num = 10000
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, BinaryClassificationEvaluator
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import sys
import argparse
from aux_models import TokenLocation, Selector
from src.data_loaders import load_data

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

SS = '[SS]'
SE = '[SE]'
OS = '[OS]'
OE = '[OE]'
tokens = ([SS, SE, OS, OE])


def valid(row):
    cond = lambda s: SS in s and OS in s
    return cond(row['s1']) and cond(row['s2'])
    if not cond(row['s1']):
        print(row['s1'])
    if not cond(row['s2']):
        print(row['s2'])
    print(row)
    return True


def df_to_samples(df):
    return [InputExample(texts=[row['s1'], row['s2']], label=float(row['l'])) for i, row in df.iterrows() if valid(row)]


def gen_all_data(rel_path):
    for rel_num in [50000, 10000, 1000, 100, 50, 20, 10]:
        print(f'extracting relations with more then {rel_num} instances')
        train, test, num_train = load_data(rel_path, sentence_path, rel_num, 99)
        print(f'got {num_train} relation types!')
        train.to_csv(f'100p_{num_train}.csv')
        break
    test.to_csv('100p_test.ftr')


def load_local_data(args):
    return pd.read_csv(args.train_path), pd.read_csv(args.test_path)

def clean_df(df):
    check = lambda s: SS in s and OS in s
    series = df.apply(lambda x: check(x['s1']) and check(x['s2']), axis=1)
    return df[series]


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default='')
    parser.add_argument("--test_path", type=str, default='')
    parser.add_argument("--device", type=str, default='all')

    return parser.parse_args()

def eval_model(model_save_path, test_samples):
    model = SentenceTransformer(model_save_path)
    test_evaluator = BinaryClassificationEvaluator.from_input_examples(test_samples, name='rel-test')
    test_evaluator(model, output_path=model_save_path)

#  --train_path train_9.ftr --test_path test1_10.csv --device 0
if __name__ == '__main__':
    # eval_model(r'F:\1_10_train_233_SpanBERT-spanbert-base-cased-2022-12-18_16-23-52')
    # eval_model(r'C:\models\1_10_train_766_SpanBERT-spanbert-base-cased-2022-12-18_16-23-52')
    # exit()
    # if 'linux' in sys.platform:
    #     dir = '/home/nlp/amirdnc/data/fs_rc/'
    #     rel_path = dir + r"rels.json"
    #     sentence_path = dir + r"text.json"
    # else:
    #     dir = r'../fewshot_RC/'
    #     rel_path = dir + r"val_rels.json"
    #     sentence_path = dir + r"val_text.json"
    # gen_all_data(rel_path)
    # exit()
    args = get_params()
    model_name = 'SpanBERT/spanbert-base-cased'  # 'microsoft/deberta-v3-base'
    # Read the dataset
    train_batch_size = 4
    num_epochs = 4
    model_save_path = 'output/1_10_'+ args.train_path.replace('.ftr', '_') +model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    # train, test = load_data(rel_path, sentence_path, rel_num)

    logging.info("Load dev dataset")
    train, test = load_local_data(args)
    word_embedding_model = models.Transformer(model_name)
    word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
    token_indexers = TokenLocation(word_embedding_model.tokenizer(f'{SS} {OS}')['input_ids'][1:-1])
    special_tokens_ids = word_embedding_model.tokenizer(f'{SS} {OS}')['input_ids'][1:-1]
    word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

    if args.device == 'all':
        device_to_train = 'cuda'
    else:
        device_to_train = 'cuda:' + args.device
    selector = Selector()
    model = SentenceTransformer(modules=[word_embedding_model,token_indexers, selector], device=device_to_train)

    train = clean_df(train)
    train_samples = df_to_samples(train)
    dev_samples = df_to_samples(test)
    test_samples = dev_samples


    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)


    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='rel-dev')
    evaluator = BinaryClassificationEvaluator.from_input_examples(dev_samples, name='rel-dev')

    # Configure the training. We skip evaluation in this example
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs  * 0.1) #10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=num_epochs,
              evaluation_steps=1000,
              warmup_steps=warmup_steps,
              output_path=model_save_path)


    ##############################################################################
    #
    # Load the stored model and evaluate its performance on STS benchmark dataset
    #
    ##############################################################################


    # eval_model(r'C:\models\50_50_train_766_SpanBERT-spanbert-base-cased-2022-12-17_20-43-13')
    eval_model(model_save_path, test_samples)