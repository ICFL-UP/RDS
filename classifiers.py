import os
import time

from joblib import parallel_backend
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier


def randomForrest(train_data, correct_class):
    with parallel_backend('threading', n_jobs=os.cpu_count()):
        start_time = time.time()
        classifier = RandomForestClassifier()
        classifier.fit(train_data, correct_class)
        print("Train time for Random Forrest: " + str((time.time() - start_time)/60) + "min")
        return classifier


def adaBoost(train_data, correct_class):
    with parallel_backend('threading', n_jobs=os.cpu_count()):
        start_time = time.time()
        classifier = AdaBoostClassifier()
        classifier.fit(train_data, correct_class)
        print("Train time for Adaboost: " + str((time.time() - start_time)/60) + "min")
        return classifier
