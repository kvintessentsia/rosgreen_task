import warnings

import joblib
from django.shortcuts import render

from .forms import TextForm

warnings.filterwarnings('ignore')
import re

import joblib
import scipy
import torch
from torch.utils.data import Dataset

TOKEN_RE = re.compile(r'[\w\d]+')


def tokenize_text_simple_regex(txt, min_token_size=4):
    txt = txt.lower()
    all_tokens = TOKEN_RE.findall(txt)
    return [token for token in all_tokens if len(token) >= min_token_size]


def character_tokenize(txt):
    return list(txt)


def tokenize_corpus(texts, tokenizer=tokenize_text_simple_regex, **tokenizer_kwargs):
    return [tokenizer(text, **tokenizer_kwargs) for text in texts]

def vectorize_texts(tokenized_texts, word2id, word2freq, mode='tfidf', scale=True):
    assert mode in {'tfidf', 'idf', 'tf', 'bin'}

    result = scipy.sparse.dok_matrix((len(tokenized_texts), len(word2id)), dtype='float32')
    for text_i, text in enumerate(tokenized_texts):
        for token in text:
            if token in word2id:
                result[text_i, word2id[token]] += 1

    if mode == 'bin':
        result = (result > 0).astype('float32')

    elif mode == 'tf':
        result = result.tocsr()
        result = result.multiply(1 / result.sum(1))


    elif mode == 'idf':
        result = (result > 0).astype('float32').multiply(1 / word2freq)

    elif mode == 'tfidf':
        result = result.tocsr()
        result = result.multiply(1 / result.sum(1))  
        result = result.multiply(1 / word2freq) 

    if scale:
        result = result.tocsc()
        result -= result.min()
        result /= (result.max() + 1e-6)

    return result.tocsr()


class SparseFeaturesDataset(Dataset):
    def __init__(self, features):
        self.features = features


    def call(self):
        return torch.from_numpy(self.features.toarray()).float()


VECTORIZATION_MODE = 'tfidf'
model = joblib.load("model.pkl")
dataset = joblib.load("test_dataset.pkl")
vocabulary = joblib.load("vocabulary.pkl")
word_doc_freq = joblib.load("word_doc_freq.pkl")
new_model = joblib.load("model_new.pkl")


def basic(request):
    form = TextForm(request.GET)
    text = request.GET.get('text')
    if text:
        text = [tokenize_text_simple_regex(text)]
        text = vectorize_texts(text, vocabulary, word_doc_freq, mode=VECTORIZATION_MODE)
        object = SparseFeaturesDataset(text).call()
        prediction = int(model(object).argmax())
        from_0_to_10 = int(new_model(object).argmax())
    else:
        prediction = 'пока нет предсказаний'
        from_0_to_10 = 'пока нет предсказаний'
    context = {
        'form': form,
        'prediction': prediction,
        'from_0_to_10': from_0_to_10,
    }
    return render(request, 'form.html', context)