import time

from gensim.models.doc2vec import TaggedDocument
from sklearn.utils import shuffle
import preprocessor
import classifiers
import data_reader
import os


def main():
    # data_reader.createStats()

    accuracies = [0.0, 0.0, 100]
    num_runs = 10
    for i in range(num_runs):
        data = data_reader.getTrainTest(250)
        # vector = preprocessor.bagOfWords(data[0][0] + data[1][0])[0].toarray()
        # vector = preprocessor.TF_IDF(data[0][0] + data[1][0])[0].toarray()
        model = preprocessor.Doc2Vec([TaggedDocument(doc, [i]) for i, doc in enumerate(data[0][0] + data[1][0])])
        vector = [model.infer_vector(x.split()) for x in data[0][0] + data[1][0]]
        train_vector = vector[:len(data[0][0])]
        test_vector = vector[len(data[0][0]):]
        shuffled_train, shuffled_classes = shuffle(train_vector, data[0][1])
        rf_classifier = classifiers.randomForrest(shuffled_train, shuffled_classes)

        shuffled_test, shuffled_classes_test = shuffle(test_vector, data[1][1])
        correct_predictions = 0
        for x in range(len(shuffled_classes_test)):
            predicted_value = rf_classifier.predict([shuffled_test[x]])[0]
            # print("Predicted Class: " + str(rf_classifier.predict([shuffled_test[x]]))
            #       + " Correct class: " + str(shuffled_classes_test[x]))
            if predicted_value == shuffled_classes_test[x]:
                correct_predictions += 1
        accuracy = (correct_predictions/len(test_vector))*100
        print("Accuracy: " + str(accuracy) + "%")
        accuracies[0] += accuracy/num_runs
        if accuracies[1] < accuracy:
            accuracies[1] = accuracy
        if accuracies[2] > accuracy:
            accuracies[2] = accuracy
    print("Overall Accuracy: " + str(accuracies[0]) + "\nBest: " + str(accuracies[1]) + "\nWorst: " + str(accuracies[2]))

    # =====> NLP <=====
    # corpus = data_reader.getCorpus()
    # # TF_IDF
    # start_time = time.time()
    # X = preprocessor.TF_IDF(corpus)[0]
    # print("Time for TF-IDF: " + str((time.time() - start_time)))
    # print(X.todense())
    # print(X.shape)
    #
    # # Bag-of-Words
    # start_time = time.time()
    # X = preprocessor.bagOfWords(corpus)[0]
    # print("Time for Bag-of-Words: " + str((time.time() - start_time)))
    # print(X.todense())
    # print(X.shape)
    #
    # # Doc2Vec
    # start_time = time.time()
    # split_corpus = [x.split() for x in corpus]
    # train_docs = [TaggedDocument(doc, [i]) for i, doc in enumerate(split_corpus[10:-10])]
    # test_docs = [TaggedDocument(doc, [i]) for i, doc in enumerate([*split_corpus[:10] + split_corpus[-10:]])]
    # model = preprocessor.dov2Vec(train_docs)
    # print("Time for Doc2Vec: " + str((time.time() - start_time)))
    #
    # num_correctly_predicted = 0
    # incorrect_predictions = []
    # for x in range(len(test_docs)):
    #     inferred_vector = model.infer_vector(test_docs[x].words)
    #     sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
    #     num_similar_benign = 0
    #     for y in sims[:300]:
    #         if y[0] <= 349:
    #             num_similar_benign += 1
    #     if x <= 349:
    #         if num_similar_benign >= 150:
    #             num_correctly_predicted += 1
    #         else:
    #             incorrect_predictions.append((x, num_similar_benign))
    #     else:
    #         if num_similar_benign < 150:
    #             num_correctly_predicted += 1
    #         else:
    #             incorrect_predictions.append((x, num_similar_benign))
    # print("Num correctly predicted: " + str(num_correctly_predicted))
    # print("Incorrect predictions: " + str(incorrect_predictions))
    # print(model.infer_vector(corpus[0].split()))


if __name__ == "__main__":
    main()
