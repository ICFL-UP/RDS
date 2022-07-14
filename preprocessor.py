import os
import time

from joblib import parallel_backend
from gensim.models.doc2vec import Doc2Vec
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def TF_IDF(train_docs):
    with parallel_backend('threading', n_jobs=os.cpu_count()):
        start_time = time.time()
        vectorizer = TfidfVectorizer(token_pattern=r"\S{2,}")
        X = vectorizer.fit_transform(train_docs)
        print("Fit time for TF-IDF: " + str(time.time() - start_time) + "sec")
        return X, vectorizer


def bagOfWords(train_docs):
    start_time = time.time()
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(train_docs)
    print("Fit time for Bag-of-Words: " + str(time.time() - start_time) + "sec")
    return X, vectorizer


def doc2Vec(train_docs):
    start_time = time.time()
    model = Doc2Vec(train_docs, window=100, min_count=2, workers=os.cpu_count())
    model.build_vocab(train_docs)
    model.train(train_docs, total_examples=model.corpus_count, epochs=model.epochs)
    print("Fit time for Doc2Vec: " + str(time.time() - start_time) + "sec")
    return model
