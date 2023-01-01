import argparse

import pandas as pd
import os
from sentence_similarity import df_to_samples, eval_model


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, default='')
    # parser.add_argument("--device", type=str, default='all')
    parser.add_argument("--model_path", type=str, default='')
    return parser.parse_args()

# --test_path "test.ftr" --model_path C:\models\50_50_train_766_SpanBERT-spanbert-base-cased-2022-12-17_20-43-13
if __name__ == '__main__':
    args = get_params()
    test = pd.read_csv(args.test_path)
    dev_samples = df_to_samples(test)
    source_dir = 'F:\output'
    for dir in os.listdir(source_dir):
        print(f'Working on {dir}')
        eval_model(os.path.join(source_dir,dir), dev_samples)