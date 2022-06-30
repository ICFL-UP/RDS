import os

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models.doc2vec import Doc2Vec


def TF_IDF(train_docs):
    vectorizer = TfidfVectorizer(token_pattern=r"\S{2,}")
    X = vectorizer.fit_transform(train_docs)
    return X, vectorizer


def bagOfWords(train_docs):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(train_docs)
    return X, vectorizer


def dov2Vec(train_docs):
    model = Doc2Vec(train_docs, window=10, min_count=1, workers=os.cpu_count())
    model.build_vocab(train_docs)
    model.train(train_docs, total_examples=model.corpus_count, epochs=model.epochs)
    return model
