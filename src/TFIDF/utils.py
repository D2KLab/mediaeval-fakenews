import csv
import pandas as pd
import numpy as np
import scipy.sparse as sp
import spacy
import re
from tqdm import tqdm
from sklearn.utils import murmurhash3_32
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def perform_SVD(texts):
    pca = PCA(n_components=3, random_state=42)
    X = pca.fit_transform(texts)
    return X


def plot_data(texts, labels):
    X_pca = perform_SVD(texts.toarray())
    df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2', 'PC3'])
    df['labels'] = labels
    sns.lmplot(x='PC1', y='PC2', data=df, fit_reg=False, hue='labels', legend=False)
    plt.legend(loc='upper right')
    plt.show()

    '''dic={1: 'blue', 2: 'orange', 3: 'green'}
    sns.set(style="darkgrid")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = df['PC1']
    y = df['PC2']
    z = df['PC3']
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.scatter(x, y, z, c=[dic[i] for i in df['labels']])
    plt.show()'''


def read_tweets(filename, task):
    ids = []
    labels = []
    tweets = []
    with open(filename, 'r', encoding="utf8") as f:
        if task == 1:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if len(row) > 3:
                    text = ''
                    for i in range(2, len(row)):
                        if i != 2:
                            text += ', ' + row[i]
                        else:
                            text += row[i]
                else:
                    text = row[2]
                ids.append(int(row[0])-1)
                labels.append(int(row[1]))
                tweets.append(text)
        elif task == 2:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if len(row) > 11:
                    text = ''
                    for i in range(10, len(row)):
                        if i != 10:
                            text += ', ' + row[i]
                        else:
                            text += row[i]
                else:
                    text = row[10]
                ids.append(int(row[0]) - 1)
                #labels.append([-1 if int(el) == 0 else 1 for el in row[1:10]])
                labels.append([int(el) for el in row[1:10]])
                tweets.append(text)
    return ids, labels, tweets


def read_tweets_text(filename):
    ids = []
    tweets = []
    with open(filename, 'r', encoding="utf8") as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if len(row) > 2:
                text = ''
                for i in range(1, len(row)):
                    if i != 1:
                        text += ', ' + row[i]
                    else:
                        text += row[i]
            else:
                text = row[1]
            ids.append(int(row[0]))
            tweets.append(text)
    return ids, tweets



def hash_f(token, num_buckets):
    """Unsigned 32 bit murmurhash for feature hashing."""
    return murmurhash3_32(token, positive=True) % num_buckets


def remove_sw(tokens):
    return [tok.text for tok in tokens if tok.is_stop is False]


def remove_puntaction(tokens):
    return [re.sub('[\W]+', '', tok.lower()) for tok in tokens]


def tokenizer_call(text):
    nlp = spacy.load('en_core_web_sm')
    tokenizer = nlp.tokenizer
    tokens = tokenizer(text)
    tokens = remove_sw(tokens)
    tokens = remove_puntaction(tokens)
    return tokens

def compute_ngrams(tokens, n):
    ngrams = [(s, e + 1)
              for s in range(len(tokens))
              for e in range(s, min(s + n, len(tokens)))]
    return ['{}'.format(' '.join(tokens[s:e])) for (s, e) in ngrams]


def count(tweets, ids, hash, n):
    row, col, val = [], [], []
    for id, tweet in zip(ids, tweets):
        tokens = tokenizer_call(str(tweet))
        ngrams = compute_ngrams(tokens, n)
        counts = Counter([hash_f(gram, hash) for gram in ngrams])
        col.extend(counts.keys())
        row.extend([id] * len(counts))
        val.extend(counts.values())
    return row, col, val


def get_count_matrix(data, n_docs, hash, ngrams):
    row, col, val = [], [], []
    step = max(int(len(n_docs) / 10), 1)
    batches = [n_docs[i:i + step] for i in range(0, len(n_docs), step)]
    for i, batch in tqdm(enumerate(batches)):
        brow, bcol, bval = count(data[batch], batch, hash, ngrams)
        row.extend(brow)
        col.extend(bcol)
        val.extend(bval)
    count_matrix = sp.csr_matrix((val, (row, col)), shape=(len(n_docs), hash))
    count_matrix.sum_duplicates()
    return count_matrix


def get_doc_freqs(cnts):
    """Return word --> # of docs it appears in."""
    binary = (cnts > 0).astype(int)
    freqs = np.array(binary.sum(1)).squeeze()
    return freqs


def compute_tfidf(data, ids, hash, ngrams):
    cnts = get_count_matrix(data, ids, hash, ngrams)
    Ns = get_doc_freqs(cnts)
    idfs = np.log((cnts.shape[1] - Ns + 0.5) / (Ns + 0.5))
    idfs[idfs < 0] = 0
    idfs = sp.diags(idfs, 0)
    tfs = cnts.log1p()
    tfidfs = idfs.dot(tfs)
    return tfidfs

def compute_tfidf_vect(data, n):
    vectorizer = TfidfVectorizer(tokenizer=tokenizer_call, lowercase=True, ngram_range=(1, n))
    tfidf_X = vectorizer.fit_transform(data)
    return tfidf_X

def compute_test_tfidf(train, text, n):
    vectorizer = TfidfVectorizer(tokenizer=tokenizer_call, lowercase=True, ngram_range=(1, n))
    X_train = vectorizer.fit_transform(train)
    X_test = vectorizer.transform(text)
    return X_train, X_test
