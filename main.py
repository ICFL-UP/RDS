import time

from gensim.models.doc2vec import TaggedDocument
import preprocessor
import data_reader
import os


def main():
    # data_reader.createStats()

    corpus = data_reader.getCorpus()
    # TF_IDF
    start_time = time.time()
    X = preprocessor.TF_IDF(corpus)[0]
    print("Time for TF-IDF: " + str((time.time() - start_time)))
    print(X.todense())
    print(X.shape)

    # Bag-of-Words
    start_time = time.time()
    X = preprocessor.bagOfWords(corpus)[0]
    print("Time for Bag-of-Words: " + str((time.time() - start_time)))
    print(X.todense())
    print(X.shape)

    # Doc2Vec
    start_time = time.time()
    split_corpus = [x.split() for x in corpus]
    train_docs = [TaggedDocument(doc, [i]) for i, doc in enumerate(split_corpus[10:-10])]
    test_docs = [TaggedDocument(doc, [i]) for i, doc in enumerate([*split_corpus[:10] + split_corpus[-10:]])]
    model = preprocessor.dov2Vec(train_docs)
    print("Time for Doc2Vec: " + str((time.time() - start_time)))

    num_correctly_predicted = 0
    incorrect_predictions = []
    for x in range(len(test_docs)):
        inferred_vector = model.infer_vector(test_docs[x].words)
        sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
        num_similar_benign = 0
        for y in sims[:300]:
            if y[0] <= 349:
                num_similar_benign += 1
        if x <= 349:
            if num_similar_benign >= 150:
                num_correctly_predicted += 1
            else:
                incorrect_predictions.append((x, num_similar_benign))
        else:
            if num_similar_benign < 150:
                num_correctly_predicted += 1
            else:
                incorrect_predictions.append((x, num_similar_benign))
    print("Num correctly predicted: " + str(num_correctly_predicted))
    print("Incorrect predictions: " + str(incorrect_predictions))
    print(model.infer_vector(corpus[0].split()))

    i = 0
    filenames = os.listdir(
        "C:\\Users\\danee\\OneDrive\\Documents\\University\\Honours\\COS 700\\Year Project\\RDS\\Data\\Strings\\")
    for x in range(len(corpus)):
        for y in range(len(corpus)):
            if x != y and corpus[x] == corpus[y]:
                print("Duplicate: " + filenames[x] + " " + filenames[y])
                i += 1
    print(i)


if __name__ == "__main__":
    main()
