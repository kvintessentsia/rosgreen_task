import os
import warnings

warnings.filterwarnings('ignore')


import joblib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dlnlputils.data import (SparseFeaturesDataset, build_vocabulary,
                             tokenize_corpus, tokenize_text_simple_regex,
                             vectorize_texts)
from dlnlputils.pipeline import (init_random_seed, predict_with_model,
                                 train_eval_loop)

init_random_seed()


train_negative = []
for filename in os.listdir(path='./aclImdb/train/neg'):
    if 'txt' not in filename:
        continue
    file = open('./aclImdb/train/neg' + '/' + filename, encoding='utf_8')
    text = file.read()
    file.close()
    train_negative.append(text)
print(1)
print(len(train_negative))

train_positive = []
for filename in os.listdir(path='./aclImdb/train/pos'):
    if 'txt' not in filename:
        continue
    file = open('./aclImdb/train/pos' + '/' + filename, encoding='utf_8')
    text = file.read()
    file.close()
    train_positive.append(text)
print(2)
print(len(train_positive))


test_negative = []
for filename in os.listdir(path='./aclImdb/test/neg'):
    if 'txt' not in filename:
        continue
    file = open('./aclImdb/test/neg' + '/' + filename, encoding='utf_8')
    text = file.read()
    file.close()
    test_negative.append(text)
print(3)

    
test_positive = []
for filename in os.listdir(path='./aclImdb/test/pos'):
    if 'txt' not in filename:
        continue
    file = open('./aclImdb/test/pos' + '/' + filename, encoding='utf_8')
    text = file.read()
    file.close()
    test_positive.append(text)
print(4)


train_negative_tokenized = tokenize_corpus(train_negative)
train_positive_tokenized = tokenize_corpus(train_positive)

test_positive_tokenized = tokenize_corpus(test_positive)
test_negative_tokenized = tokenize_corpus(test_negative)
print(test_negative_tokenized)
print(5)

train_negative_positive_tokenized = train_negative_tokenized + train_positive_tokenized
test_negative_positive_tokenized = test_negative_tokenized + test_positive_tokenized
print(6)

MAX_DF = 0.8
MIN_COUNT = 5
vocabulary, word_doc_freq = build_vocabulary(train_negative_positive_tokenized, max_doc_freq=MAX_DF, min_count=MIN_COUNT)

VECTORIZATION_MODE = 'tfidf'
train_vectors = vectorize_texts(train_negative_positive_tokenized, vocabulary, word_doc_freq, mode=VECTORIZATION_MODE)
test_vectors = vectorize_texts(test_negative_positive_tokenized, vocabulary, word_doc_freq, mode=VECTORIZATION_MODE)
print(test_vectors)
print(7)

train_neg_labels = np.zeros(np.array(train_negative).shape[0])
train_pos_labels = np.ones(np.array(train_positive).shape[0])
test_neg_labels = np.zeros(np.array(test_negative).shape[0])
test_pos_labels = np.ones(np.array(test_positive).shape[0])

train_labels = np.array(list(train_neg_labels) + list(train_pos_labels))
test_labels = np.array(list(test_neg_labels) + list(test_pos_labels))

UNIQUE_LABELS_N = len(set(train_labels))
UNIQUE_WORDS_N = len(vocabulary)

train_dataset = SparseFeaturesDataset(train_vectors, train_labels)
test_dataset = SparseFeaturesDataset(test_vectors, test_labels)

model = nn.Linear(UNIQUE_WORDS_N, UNIQUE_LABELS_N)

scheduler = lambda optim: \
    torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=5, factor=0.5, verbose=True)

best_val_loss, best_model = train_eval_loop(model=model,
                                            train_dataset=train_dataset,
                                            val_dataset=test_dataset,
                                            criterion=F.cross_entropy,
                                            lr=1e-1,
                                            epoch_n=80,
                                            batch_size=32,
                                            l2_reg_alpha=0,
                                            lr_scheduler_ctor=scheduler)




joblib.dump(test_dataset, "test_dataset.pkl")
joblib.dump(best_model, "model.pkl")
joblib.dump(vocabulary, "vocabulary.pkl")
joblib.dump(word_doc_freq, "word_doc_freq.pkl")
