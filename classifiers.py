import os
import time

from sklearn.svm import SVC
from joblib import parallel_backend
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def randomForrest(train_data, correct_class):
    with parallel_backend('threading', n_jobs=os.cpu_count()):
        start_time = time.time()
        classifier = RandomForestClassifier(n_estimators=200, criterion='log_loss')
        classifier.fit(train_data, correct_class)
        # print("Train time for Random Forrest: " + str((time.time() - start_time) / 60) + "min")
        return classifier


def adaBoost(train_data, correct_class):
    with parallel_backend('threading', n_jobs=os.cpu_count()):
        start_time = time.time()
        classifier = AdaBoostClassifier()
        classifier.fit(train_data, correct_class)
        # print("Train time for Adaboost: " + str((time.time() - start_time) / 60) + "min")
        return classifier


def svm(train_data, correct_class):
    with parallel_backend('threading', n_jobs=os.cpu_count()):
        start_time = time.time()
        classifier = SVC(kernel='sigmoid', degree=5, cache_size=400, probability=True)
        classifier.fit(train_data, correct_class)
        # print("Train time for SVM: " + str((time.time() - start_time) / 60) + "min")
        return classifier


def knn(train_data, correct_class):
    with parallel_backend('threading', n_jobs=os.cpu_count()):
        start_time = time.time()
        classifier = KNeighborsClassifier()
        classifier.fit(train_data, correct_class)
        # print("Train time for SVM: " + str((time.time() - start_time) / 60) + "min")
        return classifier


def decisionTree(train_data, correct_class):
    with parallel_backend('threading', n_jobs=os.cpu_count()):
        start_time = time.time()
        classifier = DecisionTreeClassifier()
        classifier.fit(train_data, correct_class)
        # print("Train time for SVM: " + str((time.time() - start_time) / 60) + "min")
        return classifier
