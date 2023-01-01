import json
import seaborn as sns
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import paired_cosine_distances

def get_value(counted, r1, r2):
    if r1 == r2:
        return sum([y for x, y in counted.items() if x.count(r1) >=2])
    return sum([y for x,y in counted.items() if r1 in x and r2 in x])

def draw_heatmap(df):
    df['split'] = df.apply(lambda x: eval(x['t']), axis=1)

    rels = Counter([t[0] for t in df['split']])
    rels_names = list(rels.keys())
    counted = Counter(df['t'].tolist())
    confusion_df = pd.DataFrame()

    for r1 in rels_names:
        confusion_df[r1] = pd.Series({r2: get_value(counted, r1, r2) for r2 in rels_names})
    print(confusion_df)

    sns.heatmap(confusion_df,
                # cmap='coolwarm',
                # annot=True,
                fmt='.5g')

    plt.xlabel('Predicted', fontsize=22)
    plt.ylabel('Actual', fontsize=22)
    plt.show()

if __name__ == "__main__":
    result_file = r"C:\Users\Amir\Dropbox\workspace_python\rc_fewshot\test.csv"
    # result_file = r"F:\10p_test.ftr"
    df = pd.read_csv(result_file)

    model_path = r'F:\output\50_50_train_853_SpanBERT-spanbert-base-cased-2022-12-17_20-43-13'
    model = SentenceTransformer(model_path)

    sentences = list(set(df['s1'].tolist() + df['s2'].tolist()))
    embeddings = model.encode(sentences, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    emb_dict = {sent: emb for sent, emb in zip(sentences, embeddings)}
    embeddings1 = [emb_dict[sent] for sent in df['s1'].tolist()]
    embeddings2 = [emb_dict[sent] for sent in df['s2'].tolist()]

    cosine_scores = 1 - paired_cosine_distances(embeddings1, embeddings2)
    df['s1_embd'] = pd.Series(embeddings1)
    df['s2_embd'] = pd.Series(embeddings2)
    df['score'] = pd.Series(cosine_scores)
    df.to_csv(result_file.replace('.csv', '_embd.csv').replace('ftr', '_embd.csv'))

