import time
import log
from gensim.models.doc2vec import TaggedDocument
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import RocCurveDisplay, f1_score, precision_score, classification_report, accuracy_score, recall_score
import joblib
import matplotlib.pyplot as plt
import preprocessor
import classifiers
import data_reader
import pandas as pd
from datetime import datetime
import numpy as np
import gc


def main():
    TRAIN = True
    DATA = False
    PREDICT = False

    data_filename = "12_Ransomware_Detection_using_Strings.csv"
    prefix = "12_"

    print(datetime.now())

    if DATA:
        log.log("Preparing data splitting ...")
        X, Y = data_reader.splitTrainTestVal(data_filename) 
        
        # Training Data
        log.log("\n\nPreparing TRAINING data ...")
        t_bow = preprocessor.bagOfWords(X[0])[0]
        preDoc = preprocessor.doc2Vec([TaggedDocument(doc, [i]) for i, doc in enumerate(X[0])])
        t_doc2vec = [preDoc.infer_vector(x.split()) for x in X[0]]
        t_tfidf = preprocessor.TF_IDF(X[0])[0]

        # Validation Data
        log.log("\n\nPreparing VALIDATION data ...")
        v_bow = preprocessor.bagOfWords(X[2])[0]
        preDoc = preprocessor.doc2Vec([TaggedDocument(doc, [i]) for i, doc in enumerate(X[2])])
        v_doc2vec = [preDoc.infer_vector(x.split()) for x in X[2]]
        v_tfidf = preprocessor.TF_IDF(X[2])[0]

        # Testing Data
        log.log("\n\nPreparing TESTING data ...")
        tt_bow = preprocessor.bagOfWords(X[1])[0]
        preDoc = preprocessor.doc2Vec([TaggedDocument(doc, [i]) for i, doc in enumerate(X[1])])
        tt_doc2vec = [preDoc.infer_vector(x.split()) for x in X[2]]
        tt_tfidf = preprocessor.TF_IDF(X[1])[0]

        # SAVE DATA
        # Using joblib which is similar to pickle to save data for efficiency
        log.log("\n\nSaving Data ...\n\n")
        joblib.dump(t_bow, "DATA\\Train\\"+prefix+"bow_features.pkl")
        joblib.dump(np.array(Y[0]), "DATA\\Train\\"+prefix+"bow_labels.pkl")
        joblib.dump(t_doc2vec, "DATA\\Train\\"+prefix+"doc2vec_features.pkl")
        joblib.dump(np.array(Y[0]), "DATA\\Train\\"+prefix+"doc2vec_labels.pkl")
        joblib.dump(t_tfidf, "DATA\\Train\\"+prefix+"tfidf_features.pkl")
        joblib.dump(np.array(Y[0]), "DATA\\Train\\"+prefix+"tfidf_labels.pkl")

        joblib.dump(v_bow, "DATA\\Val\\"+prefix+"bow_features.pkl")
        joblib.dump(np.array(Y[2]), "DATA\\Val\\"+prefix+"bow_labels.pkl")
        joblib.dump(v_doc2vec, "DATA\\Val\\"+prefix+"doc2vec_features.pkl")
        joblib.dump(np.array(Y[2]), "DATA\\Val\\"+prefix+"doc2vec_labels.pkl")
        joblib.dump(v_tfidf, "DATA\\Val\\"+prefix+"tfidf_features.pkl")
        joblib.dump(np.array(Y[2]), "DATA\\Val\\"+prefix+"tfidf_labels.pkl")


        joblib.dump(tt_bow, "DATA\\Test\\"+prefix+"bow_features.pkl")
        joblib.dump(np.array(Y[1]), "DATA\\Test\\"+prefix+"bow_labels.pkl")
        joblib.dump(tt_doc2vec, "DATA\\Test\\"+prefix+"doc2vec_features.pkl")
        joblib.dump(np.array(Y[1]), "DATA\\Test\\"+prefix+"doc2vec_labels.pkl")
        joblib.dump(tt_tfidf, "DATA\\Test\\"+prefix+"tfidf_features.pkl")
        joblib.dump(np.array(Y[1]), "DATA\\Test\\"+prefix+"tfidf_labels.pkl")
        

    log.log("Loading data ..")
    X = {
        "TRAIN": {
            "BOW": joblib.load("DATA\\Train\\"+prefix+"bow_features.pkl"),
            "DOC2VEC": joblib.load("DATA\\Train\\"+prefix+"doc2vec_features.pkl"),
            "TFIDF": joblib.load("DATA\\Train\\"+prefix+"tfidf_features.pkl")
        },
        "VAL": {
            "BOW": joblib.load("DATA\\Val\\"+prefix+"bow_features.pkl"),
            "DOC2VEC": joblib.load("DATA\\Val\\"+prefix+"doc2vec_features.pkl"),
            "TFIDF": joblib.load("DATA\\Val\\"+prefix+"tfidf_features.pkl")
        },
        "TEST": {
            "BOW": joblib.load("DATA\\Test\\"+prefix+"bow_features.pkl"),
            "DOC2VEC": joblib.load("DATA\\Test\\"+prefix+"doc2vec_features.pkl"),
            "TFIDF": joblib.load("DATA\\Test\\"+prefix+"tfidf_features.pkl")
        }
    }
    Y = {
        "TRAIN": {
            "BOW": joblib.load("DATA\\Train\\"+prefix+"bow_labels.pkl"),
            "DOC2VEC": joblib.load("DATA\\Train\\"+prefix+"doc2vec_labels.pkl"),
            "TFIDF": joblib.load("DATA\\Train\\"+prefix+"tfidf_labels.pkl"),
        },
        "VAL": {
            "BOW": joblib.load("DATA\\Val\\"+prefix+"bow_labels.pkl"),
            "DOC2VEC": joblib.load("DATA\\Val\\"+prefix+"doc2vec_labels.pkl"),
            "TFIDF": joblib.load("DATA\\Val\\"+prefix+"tfidf_labels.pkl"),
        },
        "TEST": {
            "BOW": joblib.load("DATA\\Test\\"+prefix+"bow_labels.pkl"),
            "DOC2VEC": joblib.load("DATA\\Test\\"+prefix+"doc2vec_labels.pkl"),
            "TFIDF": joblib.load("DATA\\Test\\"+prefix+"tfidf_labels.pkl"),
        }
    }

    if TRAIN:
        # Classifier Training
        log.log("\n\nTraining Classifiers ...")
        classifiers.randomForrest(X["TRAIN"]["BOW"], Y["TRAIN"]["BOW"], "BOW")
        classifiers.randomForrest(X["TRAIN"]["DOC2VEC"], Y["TRAIN"]["DOC2VEC"], "Doc2Vec")
        classifiers.randomForrest(X["TRAIN"]["TFIDF"], Y["TRAIN"]["TFIDF"], "TFIDF")

        # classifiers.adaBoost(X["TRAIN"]["BOW"], Y["TRAIN"]["BOW"], "BOW")
        # classifiers.adaBoost(X["TRAIN"]["DOC2VEC"], Y["TRAIN"]["DOC2VEC"], "Doc2Vec")
        # classifiers.adaBoost(X["TRAIN"]["TFIDF"], Y["TRAIN"]["TFIDF"], "TFIDF")

        # classifiers.svm(X["TRAIN"]["BOW"], Y["TRAIN"]["BOW"], "BOW")
        # classifiers.svm(X["TRAIN"]["DOC2VEC"], Y["TRAIN"]["DOC2VEC"], "Doc2Vec")
        # classifiers.svm(X["TRAIN"]["TFIDF"], Y["TRAIN"]["TFIDF"], "TFIDF")

        # classifiers.knn(X["TRAIN"]["BOW"], Y["TRAIN"]["BOW"], "BOW")
        # classifiers.knn(X["TRAIN"]["DOC2VEC"], Y["TRAIN"]["DOC2VEC"], "Doc2Vec")
        # classifiers.knn(X["TRAIN"]["TFIDF"], Y["TRAIN"]["TFIDF"], "TFIDF")

        # classifiers.decisionTree(X["TRAIN"]["BOW"], Y["TRAIN"]["BOW"], "BOW")
        # classifiers.decisionTree(X["TRAIN"]["DOC2VEC"], Y["TRAIN"]["DOC2VEC"], "Doc2Vec")
        # classifiers.decisionTree(X["TRAIN"]["TFIDF"], Y["TRAIN"]["TFIDF"], "TFIDF")

    
    if PREDICT:

        # Predict
        log.log("\n\nPREDICTING ...\n\n")
        models = {}
        for mdl in ['RF']:#, 'AB', 'SVM', 'KNN', 'DT']:
            for nlp in ['BOW', 'DOC2VEC', 'TFIDF']:
                models[mdl+"_"+nlp] = joblib.load('Models/{}_{}_model.pkl'.format(mdl, nlp))
                classifiers.evaluate_model(mdl+"_"+nlp, models[mdl+"_"+nlp], X["VAL"][nlp], Y["VAL"][nlp])










    # # data_reader.createStats()
    # overall_classes = []
    # overall_predictions = []
    # predicted_proba = [[], [], [], [], []]
    # accuracies = [0.0, 0.0, 100.0]
    # f1_scores = [0.0, 0.0, 100.0]
    # precision_scores = [0.0, 0.0, 100.0]
    # tf_predictions = [0, 0, 0, 0]  # True Positive, True Negative, False Positive, False Negative
    # num_runs = 1

    # estimators = []
    # shuffled_test_arr = [[], [], [], [], []]
    # shuffled_classes_test_arr = [[], [], [], [], []]
    # # for c_index in range(5):
    # for i in range(num_runs):
    #     start_time = time.time()
    #     # Average runs
    #     # data = data_reader.getTrainTest(0.2)
    #     # K-Fold
    #     kf = KFold(n_splits=5)
    #     data = data_reader.getTrainTestFromCSV(-1, "12_Ransomware_Detection_using_Strings.csv")

    #     for f, (train_index, test_index) in enumerate(kf.split(data[0])):

    #         split_data = ([data[0][x] for x in train_index], [data[1][x] for x in train_index]), \
    #                 ([data[0][x] for x in test_index], [data[1][x] for x in test_index])

    #         # Bag-of-words
    #         # vector = preprocessor.bagOfWords(split_data[0][0] + split_data[1][0])[0].toarray()
    #         # TF-IDF
    #         vector = preprocessor.TF_IDF(split_data[0][0] + split_data[1][0])[0].toarray()
    #         # Doc2Vec
    #         # model = preprocessor.doc2Vec([TaggedDocument(doc, [i]) for i, doc in enumerate(split_data[0][0] + split_data[1][0])])
    #         # vector = [model.infer_vector(x.split()) for x in split_data[0][0] + split_data[1][0]]

    #         train_vector = vector[:len(split_data[0][0])]
    #         test_vector = vector[len(split_data[0][0]):]
    #         shuffled_train, shuffled_classes = shuffle(train_vector, split_data[0][1])
    #         # if c_index == 0:
    #         #     # Random Forrest
    #         # classifier = classifiers.randomForrest(shuffled_train, shuffled_classes)
    #         # elif c_index == 1:
    #         #     # AdaBoost
    #         # classifier = classifiers.adaBoost(shuffled_train, shuffled_classes)
    #         # elif c_index == 2:
    #         #     # SVM
    #         # classifier = classifiers.svm(shuffled_train, shuffled_classes)
    #         # elif c_index == 3:
    #         #     # KNN
    #         classifier = classifiers.knn(shuffled_train, shuffled_classes)
    #         # elif c_index == 4:
    #         #     # Decision Tree
    #         #     classifier = classifiers.decisionTree(shuffled_train, shuffled_classes)

    #         estimators.append(classifier)
    #         shuffled_test, shuffled_classes_test = shuffle(test_vector, split_data[1][1])
    #         # shuffled_test_arr[c_index] += shuffled_test.tolist()
    #         # shuffled_classes_test_arr[c_index] += shuffled_classes_test
    #         correct_predictions = 0
    #         predicted_values = []

    #         for x in range(len(shuffled_classes_test)):
    #             predicted_value = classifier.predict([shuffled_test[x]])[0]
    #             # predicted_proba[c_index].append(classifier.predict_proba([shuffled_test[x]]))
    #             predicted_values.append(predicted_value)
    #             if predicted_value == shuffled_classes_test[x]:
    #                 correct_predictions += 1
    #                 if predicted_value == 0:
    #                     tf_predictions[1] += 1
    #                 else:
    #                     tf_predictions[0] += 1
    #             else:
    #                 if predicted_value == 0:
    #                     tf_predictions[3] += 1
    #                 else:
    #                     tf_predictions[2] += 1

    #         accuracy = (correct_predictions / len(test_vector)) * 100
    #         f1 = f1_score(shuffled_classes_test, predicted_values) * 100
    #         precision = precision_score(shuffled_classes_test, predicted_values) * 100
    #         accuracies[0] += accuracy / num_runs
    #         # accuracies_valid[0] += accuracy_valid/num_runs
    #         f1_scores[0] += f1 / num_runs
    #         # f1_scores_valid[0] += f1_valid/num_runs
    #         precision_scores[0] += precision / num_runs
    #         # precision_scores_valid[0] += precision_valid/num_runs
    #         if accuracies[1] < accuracy:
    #             accuracies[1] = accuracy
    #         if f1_scores[1] < f1:
    #             f1_scores[1] = f1
    #         if precision_scores[1] < precision:
    #             precision_scores[1] = precision

    #         if accuracies[2] > accuracy:
    #             accuracies[2] = accuracy
    #         if f1_scores[2] > f1:
    #             f1_scores[2] = f1
    #         if precision_scores[2] > precision:
    #             precision_scores[2] = precision
    #         overall_classes += shuffled_classes_test
    #         overall_predictions += predicted_values
    #         print("\n => TIME TAKEN: " + str((time.time() - start_time) / 60))

    # # fig = plt.figure(figsize=(7, 7), dpi=300)
    # # axes = fig.gca()
    # # for x in range(5):
    # #     RocCurveDisplay.from_estimator(estimators[x], shuffled_test_arr[x], shuffled_classes_test_arr[x], ax=axes)
    # # plt.show()

    # print(classification_report(overall_classes, overall_predictions, target_names=["Benign", "Malicious"], digits=4))
    # print("Average Accuracy: " + str(accuracies[0]) + "\nBest: " + str(accuracies[1]) + "\nWorst: "
    #       + str(accuracies[2]) + '\n')
    # print("Average F1: " + str(f1_scores[0]) + "\nBest: " + str(f1_scores[1]) + "\nWorst: " + str(
    #     f1_scores[2]) + '\n')
    # print("Average Precision: " + str(precision_scores[0]) + "\nBest: " + str(precision_scores[1]) + "\nWorst: "
    #       + str(precision_scores[2]))
    # print("True Positive: " + str(tf_predictions[0]) + " True Negative: " + str(tf_predictions[1])
    #       + " False positive: " + str(tf_predictions[2]) + " False negative: " + str(tf_predictions[3]) + '\n\n')


if __name__ == "__main__":
    main()
