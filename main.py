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
import traceback
from collections import Counter
import math
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import TfidfVectorizer

import warnings
warnings.filterwarnings("ignore")


def join_strings(lst):
    return " ".join(lst)


def entropy(s):
    if not s:
        return 0
    probs = Counter(s)
    total = len(s)
    return -sum((c/total)*math.log2(c/total) for c in probs.values())


def entropy_band(e):
    if e < 3.5: return "low"
    if e < 5.5: return "medium"
    return "high"


def prune_by_length(strings, min_len):
    return [s for s in strings if len(s) >= min_len]


def split_strings(s):
    if pd.isna(s):
        return []
    if "|" in s:
        return [x for x in s.split("|") if len(x) > 0]
    return [x for x in s.splitlines() if len(x) > 0]


def main():
    TRAIN = False
    DATA = False
    PREDICT = False

    data_filename = "60_Ransomware_Detection_Using_Strings_versioned.csv"
    prefix = data_filename[0:3]

    print(datetime.now())

    # Statistics
    df = pd.read_csv(data_filename)
    df["label"] = df["label"].astype(int)
    df["string_list"] = df["strings"].apply(split_strings)

    df["string_lengths"] = df["string_list"].apply(lambda lst: [len(x) for x in lst])
    df["avg_strlen"] = df["string_lengths"].apply(lambda x: np.mean(x) if x else 0)
    df["long_strings"] = df["string_lengths"].apply(lambda x: sum(l > 20 for l in x))

    df.groupby("label")[["avg_strlen", "long_strings"]].mean()
    print(df.groupby("label")[["avg_strlen", "long_strings"]].mean())
    print("\n\n")
    print(df.head())

    # Entropy
    df["entropies"] = df["string_list"].apply(lambda lst: [entropy(s) for s in lst])
    df["avg_entropy"] = df["entropies"].apply(lambda x: np.mean(x) if x else 0)

    df.groupby("label")["avg_entropy"].mean()
    df["entropy_band"] = df["entropies"].apply(
        lambda lst: tuple(entropy_band(e) for e in lst)
    )
    print("Entropy Stats:")
    print(df.groupby("label")["avg_entropy"].mean())


    print("\n\n")

    # Noise
    for L in [4, 8, 12, 16]:
        df[f"strings_L{L}"] = df["string_list"].apply(
            lambda x: prune_by_length(x, L)
        )

    print(df.groupby("label")[[f"strings_L{L}" for L in [4, 8, 12, 16]]].agg(
        lambda lst: np.mean([len(x) for x in lst])
    ))

    # Filter
    all_strings = Counter(s for lst in df["string_list"] for s in lst)
    top_common = set([s for s, _ in all_strings.most_common(100)])

    df["strings_pruned_common"] = df["string_list"].apply(
        lambda lst: [s for s in lst if s not in top_common]
    )

    # Mutual Information
    X_text = df["string_list"].apply(join_strings)
    y = df["label"]

    vectorizer = TfidfVectorizer(
        min_df=5,
        max_df=0.8,
        token_pattern=r"[A-Za-z0-9_\-\.]{4,}"
    )

    X = vectorizer.fit_transform(X_text)
    mi = mutual_info_classif(X, y, discrete_features=True)

    mi_scores = pd.DataFrame({
        "string": vectorizer.get_feature_names_out(),
        "MI": mi
    }).sort_values("MI", ascending=False)

    print(mi_scores.head(20))

    # # Balance Dataset
    # df = pd.read_csv(data_filename)
    # df = df.groupby('label')
    # df = df.apply(lambda x: x.sample(df.size().min()).reset_index(drop=True))
    # df.to_csv("B"+data_filename)

    # # STATS 
    data = pd.read_csv(data_filename)
    log.log("Dataset stats - Category: " + str(data["category"].value_counts()))
    log.log("Dataset stats - Label: " + str(data["label"].value_counts()))
    log.log("Dataset stats - shape: " + str(data.shape))

    if DATA:
        log.log("Preparing data splitting ...")
        data_reader.splitTrainTestVal(data_filename) 

    log.log("Loading data ..")
    X = {
        "TRAIN": {
            "BOW": joblib.load("DATA/Train/"+prefix+"bow_features.pkl"),
            "DOC2VEC": joblib.load("DATA/Train/"+prefix+"doc2vec_features.pkl"),
            "TFIDF": joblib.load("DATA/Train/"+prefix+"tfidf_features.pkl")
        },
        "VAL": {
            "BOW": joblib.load("DATA/Val/"+prefix+"bow_features.pkl"),
            "DOC2VEC": joblib.load("DATA/Val/"+prefix+"doc2vec_features.pkl"),
            "TFIDF": joblib.load("DATA/Val/"+prefix+"tfidf_features.pkl")
        },
        "TEST": {
            "BOW": joblib.load("DATA/Test/"+prefix+"bow_features.pkl"),
            "DOC2VEC": joblib.load("DATA/Test/"+prefix+"doc2vec_features.pkl"),
            "TFIDF": joblib.load("DATA/Test/"+prefix+"tfidf_features.pkl")
        }
    }
    Y = {
        "TRAIN": {
            "BOW": joblib.load("DATA/Train/"+prefix+"bow_labels.pkl"),
            "DOC2VEC": joblib.load("DATA/Train/"+prefix+"doc2vec_labels.pkl"),
            "TFIDF": joblib.load("DATA/Train/"+prefix+"tfidf_labels.pkl"),
        },
        "VAL": {
            "BOW": joblib.load("DATA/Val/"+prefix+"bow_labels.pkl"),
            "DOC2VEC": joblib.load("DATA/Val/"+prefix+"doc2vec_labels.pkl"),
            "TFIDF": joblib.load("DATA/Val/"+prefix+"tfidf_labels.pkl"),
        },
        "TEST": {
            "BOW": joblib.load("DATA/Test/"+prefix+"bow_labels.pkl"),
            "DOC2VEC": joblib.load("DATA/Test/"+prefix+"doc2vec_labels.pkl"),
            "TFIDF": joblib.load("DATA/Test/"+prefix+"tfidf_labels.pkl"),
        }
    }

    log.log("Train BOW" + str(data_reader.stats(X["TRAIN"]["BOW"], Y["TRAIN"]["BOW"])))
    log.log("Train DOC2VEC" + str(data_reader.stats(np.array(X["TRAIN"]["DOC2VEC"]), Y["TRAIN"]["DOC2VEC"])))
    log.log("Train TFIDF" + str(data_reader.stats(X["TRAIN"]["TFIDF"], Y["TRAIN"]["TFIDF"])))

    log.log("VAL BOW" + str(data_reader.stats(X["VAL"]["BOW"], Y["VAL"]["BOW"])))
    log.log("VAL DOC2VEC" + str(data_reader.stats(np.array(X["VAL"]["DOC2VEC"]), Y["VAL"]["DOC2VEC"])))
    log.log("VAL TFIDF" + str(data_reader.stats(X["VAL"]["TFIDF"], Y["VAL"]["TFIDF"])))

    log.log("TEST BOW" + str(data_reader.stats(X["TEST"]["BOW"], Y["TEST"]["BOW"])))
    log.log("TEST DOC2VEC" + str(data_reader.stats(np.array(X["TEST"]["DOC2VEC"]), Y["TEST"]["DOC2VEC"])))
    log.log("TEST TFIDF" + str(data_reader.stats(X["TEST"]["TFIDF"], Y["TEST"]["TFIDF"])))

    if TRAIN:
        # Classifier Training
        log.log("\n\nTraining Classifiers ...")
        try:
            classifiers.randomForrest(X["TRAIN"]["BOW"], Y["TRAIN"]["BOW"], "BOW")
            classifiers.randomForrest(X["TRAIN"]["DOC2VEC"], Y["TRAIN"]["DOC2VEC"], "Doc2Vec")
            classifiers.randomForrest(X["TRAIN"]["TFIDF"], Y["TRAIN"]["TFIDF"], "TFIDF")
        except:
            print(traceback.print_exc())
            log.log("\n\n\n\n\nERROR in training of RF\n\n\n\n")

        try:
            classifiers.adaBoost(X["TRAIN"]["BOW"], Y["TRAIN"]["BOW"], "BOW")
            classifiers.adaBoost(X["TRAIN"]["DOC2VEC"], Y["TRAIN"]["DOC2VEC"], "Doc2Vec")
            classifiers.adaBoost(X["TRAIN"]["TFIDF"], Y["TRAIN"]["TFIDF"], "TFIDF")
        except:
            print(traceback.print_exc())
            log.log("\n\n\n\n\nERROR in training of AB\n\n\n\n")

        try:
            classifiers.svm(X["TRAIN"]["BOW"], Y["TRAIN"]["BOW"], "BOW")
            classifiers.svm(X["TRAIN"]["DOC2VEC"], Y["TRAIN"]["DOC2VEC"], "Doc2Vec")
            classifiers.svm(X["TRAIN"]["TFIDF"], Y["TRAIN"]["TFIDF"], "TFIDF")
        except:
            print(traceback.print_exc())
            log.log("\n\n\n\n\nERROR in training of SVM\n\n\n\n")

        try:
            classifiers.knn(X["TRAIN"]["BOW"], Y["TRAIN"]["BOW"], "BOW")
            classifiers.knn(X["TRAIN"]["DOC2VEC"], Y["TRAIN"]["DOC2VEC"], "Doc2Vec")
            classifiers.knn(X["TRAIN"]["TFIDF"], Y["TRAIN"]["TFIDF"], "TFIDF")
        except:
            print(traceback.print_exc())
            log.log("\n\n\n\n\nERROR in training of KNN\n\n\n\n")

        try:
            classifiers.decisionTree(X["TRAIN"]["BOW"], Y["TRAIN"]["BOW"], "BOW")
            classifiers.decisionTree(X["TRAIN"]["DOC2VEC"], Y["TRAIN"]["DOC2VEC"], "Doc2Vec")
            classifiers.decisionTree(X["TRAIN"]["TFIDF"], Y["TRAIN"]["TFIDF"], "TFIDF")
        except:
            print(traceback.print_exc())   
            log.log("\n\n\n\n\nERROR in training of DT\n\n\n\n")

    if PREDICT:

        # Predict
        log.log("\n\nPREDICTING ...\n\n")
        models = {}
        for mdl in ['DT', 'RF', 'AB', 'SVM', 'KNN']:
            for nlp in ['BOW', 'TFIDF', 'DOC2VEC']:
                models[mdl+"_"+nlp] = joblib.load('Models/{}_{}_model.pkl'.format(mdl, nlp))
                classifiers.evaluate_model(mdl+"_"+nlp, models[mdl+"_"+nlp], X["VAL"][nlp], Y["VAL"][nlp])

        ##ROC Curve
        # for nlp in ['BOW', 'TFIDF', 'DOC2VEC']:
        #     fig = plt.figure(figsize=(7, 7), dpi=300)
        #     axes = fig.gca()
        #     for x in  ['DT', 'RF', 'AB', 'SVM', 'KNN']:
        #         RocCurveDisplay.from_estimator(models[x+"_"+nlp], X["VAL"][nlp], Y["VAL"][nlp], ax=axes)
        #     # plt.show()
        #     plt.safe(nlp+".png")

        # # BEST MODELS
        # classifiers.evaluate_model("AB_TFIDF", joblib.load('Models/AB_TFIDF_model.pkl'), X["TEST"]["TFIDF"], Y["TEST"]["TFIDF"])
        # classifiers.evaluate_model("SVM_BOW", joblib.load('Models/SVM_BOW_model.pkl'), X["TEST"]["BOW"], Y["TEST"]["BOW"])
        # classifiers.evaluate_model("DT_TFIDF", joblib.load('Models/DT_TFIDF_model.pkl'), X["TEST"]["TFIDF"], Y["TEST"]["TFIDF"])


if __name__ == "__main__":
    main()
