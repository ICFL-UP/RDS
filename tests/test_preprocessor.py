import random
import unittest
import numpy as np

from gensim.models.doc2vec import TaggedDocument
import data_reader
import preprocessor

corpus = data_reader.getCorpus()
tagged_corpus = [TaggedDocument(doc, [i]) for i, doc in enumerate([x.split() for x in corpus])]


class TestTfIdf(unittest.TestCase):

    def test_notEmpty(self):
        X, vectorizer = preprocessor.TF_IDF(corpus)
        dense_X = X.toarray()
        allZero = True
        for x in dense_X:
            for y in x:
                if y != 0:
                    allZero = False
                    break
        self.assertFalse(allZero)


class TestBagOfWords(unittest.TestCase):

    def test_notEmpty(self):
        X, vectorizer = preprocessor.bagOfWords(corpus)
        dense_X = X.toarray()
        allZero = True
        for x in dense_X:
            for y in x:
                if y != 0:
                    allZero = False
                    break
        self.assertFalse(allZero)


class TestDoc2Vec(unittest.TestCase):

    def test_notEmpty(self):
        model = preprocessor.dov2Vec(tagged_corpus)
        vector = model.infer_vector(random.choice(corpus).split())
        allZero = True
        for x in vector:
            if x != 0:
                allZero = False
                break
        self.assertFalse(allZero)


if __name__ == '__main__':
    unittest.main()
