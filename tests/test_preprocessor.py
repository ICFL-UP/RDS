import random
import unittest
import numpy as np

from gensim.models.doc2vec import TaggedDocument
import data_reader
import preprocessor

corpus = data_reader.getCorpus()
tagged_corpus = [TaggedDocument(doc, [i]) for i, doc in enumerate([x.split() for x in corpus])]


class TestTfIdf(unittest.TestCase):

    def test_distinctVectors(self):
        X, vectorizer = preprocessor.TF_IDF(corpus)
        dense_X = X.todense()
        for x in range(len(dense_X)):
            for y in range(len(dense_X)):
                if x != y:
                    if np.array_equal(dense_X[x], dense_X[y]):
                        print(x)
                        print(y)
                    self.assertFalse(np.array_equal(dense_X[x], dense_X[y]))


class TestBagOfWords(unittest.TestCase):

    def test_distinctVectors(self):
        X, vectorizer = preprocessor.bagOfWords(corpus)
        dense_X = X.todense()
        for x in range(len(dense_X)):
            for y in range(len(dense_X)):
                if x != y:
                    self.assertFalse(np.array_equal(dense_X[x], dense_X[y]))


class TestDoc2Vec(unittest.TestCase):

    def test_distinctVectors(self):
        model = preprocessor.dov2Vec(tagged_corpus)
        vec_one = model.infer_vector(random.choice(corpus).split())
        vec_two = model.infer_vector(random.choice(corpus).split())
        self.assertFalse(np.array_equal(vec_one, vec_two))

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
