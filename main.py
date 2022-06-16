from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import common_texts
import data_reader
import collections
import random
import os


def main():
    # data_reader.createStats()

    # TF_IDF
    corpus = []
    filenames = os.listdir('Data/Strings')
    for x in filenames:
        file = open("Data/Strings/" + x, 'r')
        document = ''
        for y in file.readlines():
            document += y.replace('\n', '') + ' '
        corpus.append(document)
    vectorizer = TfidfVectorizer(token_pattern=r"\S{2,}")
    X = vectorizer.fit_transform(corpus)
    # print(X)

    # Bag-of-Words
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    # print(vectorizer.get_feature_names_out().tolist())

    # Doc2Vec
    split_corpus = [x.split() for x in corpus]
    train_docs = [TaggedDocument(doc, [i]) for i, doc in enumerate([*split_corpus[10:] + split_corpus[:-10]])]
    test_docs = [TaggedDocument(doc, [i]) for i, doc in enumerate([*split_corpus[:10] + split_corpus[-10:]])]
    model = Doc2Vec(train_docs, window=10, min_count=1, workers=4)
    model.build_vocab(train_docs)
    model.train(train_docs, total_examples=model.corpus_count, epochs=model.epochs)

    ranks = []
    second_ranks = []
    for doc_id in range(len(train_docs)):
        inferred_vector = model.infer_vector(train_docs[doc_id].words)
        sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)
        second_ranks.append(sims[1])

        if doc_id == 0:
            print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_docs[doc_id].words)))
            print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
            for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
                print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_docs[sims[index][0]].words)))

    print(collections.Counter(ranks))

    doc_id = random.randint(0, len(test_docs) - 1)
    inferred_vector = model.infer_vector(test_docs[doc_id].words)
    sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))

    print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_docs[doc_id].words)))
    print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
    for label, index in [('MOST', 0), ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
        print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_docs[sims[index][0]].words)))

    print(model.infer_vector(split_corpus[0]))


if __name__ == "__main__":
    main()
