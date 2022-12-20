import log

import os
import time

from joblib import parallel_backend
from gensim.models.doc2vec import Doc2Vec
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def TF_IDF(train_docs):
    log.log("Processing TF-IDF ...")
    with parallel_backend('threading', n_jobs=os.cpu_count()):
        start_time = time.time()
        vectorizer = TfidfVectorizer(token_pattern=r"\S{2,}")
        X = vectorizer.fit_transform(train_docs)
        log.log("Fit time for TF-IDF: " + str((time.time() - start_time) / 60) + " min")
        return X, vectorizer


def bagOfWords(train_docs):
    log.log("Processing Bag-ofWords ...")
    start_time = time.time()
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(train_docs)
    log.log("Fit time for Bag-of-Words: " + str((time.time() - start_time) / 60) + " min")
    return X, vectorizer


def doc2Vec(train_docs):
    log.log("Processing Doc2Vec ...")
    start_time = time.time()
    model = Doc2Vec(train_docs, vector_size=1000, window=50, min_count=1, dm=0, workers=os.cpu_count())
    model.build_vocab(train_docs)
    model.train(train_docs, total_examples=model.corpus_count, epochs=model.epochs)
    log.log("Fit time for Doc2Vec: " + str((time.time() - start_time) / 60) + " min")
    return model
