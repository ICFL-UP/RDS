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
import os
import glob
import rl_env
import rl_agent
import torch


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


def moving_average(x, w=5):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w), 'valid') / w


def main():

    TRAIN = False
    DATA = False
    PREDICT = True
    STATS = False
    RL = True
    RL_EPISODES = 30

    data_filename = "60_Ransomware_Detection_Using_Strings_versioned.csv"
    prefix = data_filename[0:3]

    print(datetime.now())

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

    if PREDICT:

        log.log("\n\nPREDICTING ...\n\n")

        models = {}
        for mdl in ['DT','RF','AB','SVM','KNN']:
            for nlp in ['BOW','TFIDF','DOC2VEC']:
                models[mdl+"_"+nlp] = joblib.load('Models/{}_{}_model.pkl'.format(mdl,nlp))
                classifiers.evaluate_model(mdl+"_"+nlp, models[mdl+"_"+nlp], X["VAL"][nlp], Y["VAL"][nlp])

    if RL:

        log.log("\n\nRUNNING RL WORKFLOW\n\n")

        candidate_models = {}
        for path in glob.glob('Models/*_model.pkl'):
            name = os.path.basename(path).replace('_model.pkl','')
            try:
                candidate_models[name] = joblib.load(path)
            except Exception:
                log.log(f"Failed to load {path}")

        best_name = None
        best_score = -1

        for name,mdl in candidate_models.items():

            nlp = name.split('_')[-1]
            Xval = X['VAL'][nlp]
            yval = np.array(Y['VAL'][nlp])

            try:

                Xval_dense = Xval.toarray() if hasattr(Xval,'toarray') else np.array(Xval)
                pred = mdl.predict(Xval_dense)
                sc = f1_score(yval,pred)

                log.log(f"Model {name} F1 on VAL: {sc}")

                if sc > best_score:

                    best_score = sc
                    best_name = name
                    best_model = mdl
                    best_nlp = nlp

            except Exception as e:
                log.log(f"Skipping {name}: {e}")

        if best_name is None:
            log.log("No supervised models found for RL initialization.")
            return

        log.log(f"Selected best model: {best_name} (F1={best_score})")

        Xtrain = X['TRAIN'][best_nlp]
        Ytrain = np.array(Y['TRAIN'][best_nlp]).astype(int)

        Xtrain = Xtrain.toarray() if hasattr(Xtrain,'toarray') else np.array(Xtrain)

        input_dim = Xtrain.shape[1]

        policy = rl_agent.PolicyNetwork(input_dim)

        try:

            y_boot = best_model.predict(Xtrain)
            rl_agent.behavioral_clone(policy,Xtrain,y_boot,epochs=5,lr=1e-3)

            log.log("Behavioral cloning completed.")

        except Exception as e:
            log.log(f"Behavioral cloning failed: {e}")

        env = rl_env.RLDatasetEnv(Xtrain,Ytrain)

        episode_rewards = []
        episode_lengths = []

        try:

            episode_rewards, episode_lengths = rl_agent.reinforce_train(
                policy,
                env,
                episodes=RL_EPISODES,
                lr=1e-3,
                gamma=0.99,
                track_metrics=True
            )

            log.log("RL training finished.")

        except Exception as e:
            log.log(f"RL training failed: {e}")

        # Plot reward curve

        plt.figure(figsize=(8,5))
        plt.plot(episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("RL Training Reward Curve")
        plt.grid(True)
        plt.savefig("RL_reward_curve.png",dpi=300)
        plt.close()

        # Smoothed reward

        plt.figure(figsize=(8,5))
        plt.plot(moving_average(episode_rewards))
        plt.xlabel("Episode")
        plt.ylabel("Smoothed Reward")
        plt.title("Smoothed RL Reward Convergence")
        plt.grid(True)
        plt.savefig("RL_reward_smoothed.png",dpi=300)
        plt.close()

        Xtest = X['TEST'][best_nlp]
        Ytest = np.array(Y['TEST'][best_nlp]).astype(int)

        Xtest = Xtest.toarray() if hasattr(Xtest,'toarray') else np.array(Xtest)

        try:

            ypred = rl_agent.policy_predict(policy,Xtest)

            report = classification_report(Ytest,ypred)
            log.log(f"RL policy evaluation on TEST:\n{report}")

        except Exception as e:
            log.log(f"Failed to evaluate RL policy: {e}")
            return

        # Confusion Matrix

        from sklearn.metrics import ConfusionMatrixDisplay

        fig,ax = plt.subplots(figsize=(6,6))
        ConfusionMatrixDisplay.from_predictions(Ytest,ypred,ax=ax)
        plt.title("RL Policy Confusion Matrix")
        plt.savefig("RL_confusion_matrix.png",dpi=300)
        plt.close()

        # Baseline comparison

        baseline_pred = best_model.predict(Xtest)

        rl_f1 = f1_score(Ytest,ypred)
        rl_acc = accuracy_score(Ytest,ypred)

        baseline_f1 = f1_score(Ytest,baseline_pred)
        baseline_acc = accuracy_score(Ytest,baseline_pred)

        comparison = pd.DataFrame({
            "Model":["Baseline_"+best_name,"RL_Policy"],
            "F1":[baseline_f1,rl_f1],
            "Accuracy":[baseline_acc,rl_acc]
        })

        print(comparison)

        comparison.to_csv("RL_vs_Supervised.csv",index=False)


if __name__ == "__main__":
    main()
