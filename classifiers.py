import log

import os
import time
import numpy as np
from sklearn.svm import SVC
from joblib import parallel_backend 
import joblib
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss,  confusion_matrix

def print_results(results, classifier):
    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        log.log('{} (+/-{}) for {}'.format(round(mean, 4), round(std * 2, 4), params))
    log.log('BEST PARAMS for {}: {}\n'.format(classifier, results.best_params_))


def evaluate_model(name, model, features, labels):
    start = time.time()
    pred = model.predict(features)
    end = time.time()
    accuracy = round(accuracy_score(labels, pred), 4)
    precision = round(precision_score(labels, pred, pos_label='M'), 4)
    recall = round(recall_score(labels, pred, pos_label='M'), 4)
    f1 = round(f1_score(labels, pred, pos_label='M'), 4)
    auc = round(roc_auc_score(labels, model.predict_proba(features)[:, 1]), 4)
    logloss = round(log_loss(labels, model.predict_proba(features)), 4)

    cm = confusion_matrix(labels, pred)
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    lat = round((end - start)*1000, 4)
    log.log('\n\n{} -- TP: {} / TN: {} / FP: {} / FN: {} / Recall: {} / Precision: {} / F1-Score: {} / Accuracy: {}  / AUC: {} / LogLoss: {} / Latency: {}ms / Num: {}'.format(
        name, TP[0], TN[0], FP[0], FN[0], recall, precision, f1, accuracy, auc, logloss, lat, len(pred)))
    log.log("{} & {} & {} & {} & {} & {} & {} \n{} & {} & {} & {} & {}".format(TP[0], TN[0], FP[0], FN[0], recall, precision, f1, accuracy, auc, logloss, lat, round(lat/len(pred),4)))


def randomForrest(train_data, correct_class, nlp):
    log.log("\nTraining RandomForrest for {}...".format(nlp))
    with parallel_backend('threading', n_jobs=os.cpu_count()):
        
        start_time = time.time()
        rf = RandomForestClassifier()
        param = {
            'n_estimators': [5, 50, 250],
            'max_depth': [2, 4, 8, 16, 32, None]
        }

        cv = GridSearchCV(rf, param, cv=5)
        cv.fit(train_data, correct_class.ravel())
        print_results(cv, "RandomForrest + {}".format(nlp))
        joblib.dump(cv.best_estimator_, 'Models/RF_{}_model.pkl'.format(nlp))
        log.log("Train time for Random Forrest + {}: ".format(nlp) + str((time.time() - start_time) / 60) + " min")
        return cv.best_estimator_


def adaBoost(train_data, correct_class, nlp):
    log.log("\nTraining AdaBoost for {}...".format(nlp))
    with parallel_backend('threading', n_jobs=os.cpu_count()):
        start_time = time.time()
        classifier = AdaBoostClassifier()
        param = {
            'n_estimators': [5, 50, 250],
            'learning_rate': [0.01, 0.1, 1, 10, 100]
        }

        cv = GridSearchCV(classifier, param, cv=5)
        cv.fit(train_data, correct_class.ravel())
        print_results(cv, "AdaBoost + {}".format(nlp))
        joblib.dump(cv.best_estimator_, 'Models/AB_{}_model.pkl'.format(nlp))
        log.log("Train time for AdaBoost + {}: ".format(nlp) + str((time.time() - start_time) / 60) + " min")
        return cv.best_estimator_


def svm(train_data, correct_class, nlp):
    log.log("\nTraining SVM for {}...".format(nlp))
    with parallel_backend('threading', n_jobs=os.cpu_count()):
        start_time = time.time()
        classifier = SVC()  
        param = {
            'kernel': ['linear', 'rbg', 'sigmoid'],
            'C': [0.1, 1, 10],
            'probability': 'true'
        }

        cv = GridSearchCV(classifier, param, cv=5)
        cv.fit(train_data, correct_class.ravel())
        print_results(cv, "SVM + {}".format(nlp))
        joblib.dump(cv.best_estimator_, 'Models/SVM_{}_model.pkl'.format(nlp))
        log.log("Train time for SVM + {}: ".format(nlp) + str((time.time() - start_time) / 60) + " min")
        return cv.best_estimator_


def knn(train_data, correct_class, nlp):
    log.log("\nTraining KNN for {}...".format(nlp))
    with parallel_backend('threading', n_jobs=os.cpu_count()):
        start_time = time.time()
        classifier = KNeighborsClassifier()       
        param = {
            'n_neighbors': [5, 50, 250],
            'weights': ['uniform', 'distance'],
            'p': [1, 2],
            'leaf_size':  [5, 50, 250]
        }

        cv = GridSearchCV(classifier, param, cv=5)
        cv.fit(train_data, correct_class.ravel())
        print_results(cv, "KNN + {}".format(nlp))
        joblib.dump(cv.best_estimator_, 'Models/KNN_{}_model.pkl'.format(nlp))
        log.log("Train time for KNN + {}: ".format(nlp) + str((time.time() - start_time) / 60) + " min")
        return cv.best_estimator_


def decisionTree(train_data, correct_class, nlp):
    log.log("\nTraining Decision Tree for {}...".format(nlp))
    with parallel_backend('threading', n_jobs=os.cpu_count()):
        start_time = time.time()
        classifier = DecisionTreeClassifier()
        param = {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'splitter': ['best', 'random']
        }

        cv = GridSearchCV(classifier, param, cv=5)
        cv.fit(train_data, correct_class.ravel())
        print_results(cv, "DT + {}".format(nlp))
        joblib.dump(cv.best_estimator_, 'Models/DT_{}_model.pkl'.format(nlp))
        log.log("Train time for DT + {}: ".format(nlp) + str((time.time() - start_time) / 60) + " min")
        return cv.best_estimator_
