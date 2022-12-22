import log

import json
import os
import random
from sklearn.model_selection import train_test_split
import pandas as pd
import preprocessor

def createStats():
    benign_strings_compendium = []
    malicious_strings_compendium = []
    filenames = [file for file in os.listdir('Strings\\') if file.endswith('.txt')]
    benign_most_strings = 0
    benign_least_strings = float('inf')
    benign_total_num_strings = 0
    malicious_most_strings = 0
    malicious_least_strings = float('inf')
    malicious_total_num_strings = 0
    num_benign_files = 0
    num_malicious_files = 0
    for x in filenames:
        file = open("Strings\\" + x, 'r')
        lines = file.readlines()
        lines_length = len(lines)
        if x.startswith('B_'):
            num_benign_files += 1
            benign_strings_compendium += lines
            benign_total_num_strings += lines_length
            if lines_length < benign_least_strings:
                benign_least_strings = lines_length
            if lines_length > benign_most_strings:
                benign_most_strings = lines_length
        else:
            num_malicious_files += 1
            malicious_strings_compendium += lines
            malicious_total_num_strings += lines_length
            if lines_length < malicious_least_strings:
                malicious_least_strings = lines_length
            if lines_length > malicious_most_strings:
                malicious_most_strings = lines_length

    duplicates = 0
    num_files = len(filenames)
    average_num_strings = (benign_total_num_strings + malicious_total_num_strings) / num_files
    unique_strings_total = []
    unique_strings_benign = []
    unique_strings_malicious = []
    num_its = 0
    for x in benign_strings_compendium:
        print("On iteration: " + str(num_its))
        if x not in unique_strings_total:
            unique_strings_total.append(x)
        else:
            duplicates += 1
        if x not in unique_strings_benign:
            unique_strings_benign.append(x)
        num_its += 1
    for x in malicious_strings_compendium:
        print("On iteration: " + str(num_its))
        if x not in unique_strings_total:
            unique_strings_total.append(x)
        else:
            duplicates += 1
        if x not in unique_strings_malicious:
            unique_strings_malicious.append(x)
        num_its += 1

    file = open('Stats.txt', 'w')
    file.write("Number of files: " + str(num_files) + "\nNumber of benign files: " + str(num_benign_files)
               + "\nNumber of malicious files: " + str(num_malicious_files)
               + "\nTotal number of strings: " + str(benign_total_num_strings + malicious_total_num_strings)
               + "\nNumber of benign strings: " + str(benign_total_num_strings)
               + "\nNumber of malicious strings: " + str(malicious_total_num_strings)
               + "\nTotal-average number of strings: " + str(average_num_strings)
               + "\nAverage number of strings benign: " + str(benign_total_num_strings / num_benign_files)
               + "\nAverage number of strings malicious: "
               + str(malicious_total_num_strings / num_malicious_files)
               + "\nLeast strings benign: " + str(benign_least_strings) + " Most: " + str(benign_most_strings)
               + "\nLeast strings malicious: " + str(malicious_least_strings) + " Most: " + str(malicious_most_strings)
               + "\nNumber of unique strings in total: " + str(len(unique_strings_total))
               + "\n - Of which benign: " + str(len(unique_strings_benign))
               + "\n - Of which malicious: " + str(len(unique_strings_malicious)))
    file.close()


# def getStrings(file=None):
#     all_strings = []
#     if file is None:
#         file_names = [file for file in os.listdir('Data\\') if file.endswith('.json')]
#     else:
#         file_names = [file]
#     for i in file_names:
#         temp_string = ""
#         file = open("Data\\" + i, 'r')
#         try:
#             file_strings = json.loads(file.read())['strings']
#             for x in file_strings:
#                 temp_string += x + ' '
#             all_strings.append(temp_string)
#         except json.decoder.JSONDecodeError:
#             print("File: " + i + " FAILED!")
#         except KeyError:
#             print("Key error in file: " + i)
#     return all_strings

def getStrings():
    benign_strings = {}
    malicious_strings = {}
    benign_file_names = [f for f in os.listdir("Strings\\") if f.startswith('B_')]
    malicious_file_names = [f for f in os.listdir("Strings\\") if f.startswith('M_')]
    for x in benign_file_names:
        nested_dict = {}
        benign_strings[x] = nested_dict
        file = open("Strings\\" + x, 'r')
        all_lines = file.readlines()
        for line in all_lines:
            cleaned_line = line.replace('\n', ' ')
            if cleaned_line not in nested_dict:
                nested_dict[cleaned_line] = 1
            else:
                nested_dict[cleaned_line] += 1
        nested_dict['total'] = len(all_lines)
        file.close()
    for x in malicious_file_names:
        nested_dict = {}
        malicious_strings[x] = nested_dict
        file = open("Strings\\" + x, 'r')
        all_lines = file.readlines()
        for line in all_lines:
            cleaned_line = line.replace('\n', ' ')
            if cleaned_line not in nested_dict:
                nested_dict[cleaned_line] = 1
            else:
                nested_dict[cleaned_line] += 1
        nested_dict['total'] = len(all_lines)
        file.close()
    return benign_strings, malicious_strings


def writeStrings(split=False):
    file = open('Strings.txt', 'w')
    file_names = [file for file in os.listdir('Data\\') if file.endswith('.json')]
    for x in file_names:
        temp_file = open("Data\\" + x, 'r')
        try:
            strs = json.loads(temp_file.read())['strings']
            if not split:
                for i in range(len(strs)):
                    file.write(strs[i] + '\n')
            else:
                split_file = open("Strings\\" + x.replace('.json', '') + '.txt', 'w')
                for i in range(len(strs)):
                    split_file.write(strs[i] + '\n')
                split_file.close()
        except json.decoder.JSONDecodeError:
            print("File: " + x + " FAILED!")
        except KeyError:
            print("Key error in file: " + x)
        temp_file.close()
    file.close()


def getCorpus():
    corpus = []
    filenames = os.listdir("Strings")
    for x in filenames:
        file = open("Strings\\" + x,'r')
        document = ''
        for y in file.readlines():
            document += y.replace('\n', ' ')
        corpus.append(document)
    return corpus


def getTrainTest(percent_test):
    print("Reading Strings from files ...")
    benign_strings = []
    malicious_strings = []
    benign_file_names = [f for f in os.listdir("Strings") if f.startswith('B_')]
    malicious_file_names = [f for f in os.listdir("Strings") if f.startswith('M_')]
    for x in benign_file_names:
        file = open("Strings\\" + x, 'r')
        all_lines = file.readlines()
        benign_strings.append(' '.join(all_lines))
        file.close()
    for x in malicious_file_names:
        file = open("Strings\\" + x, 'r')
        all_lines = file.readlines()
        malicious_strings.append(' '.join(all_lines))
        file.close()

    X = benign_strings + malicious_strings
    y = [0] * len(benign_strings) + [1] * len(malicious_strings)

    if percent_test == -1:
        return X, y

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=percent_test)
    return (X_train, y_train), (X_test, y_test)


def splitTrainTestVal(filename):
    log.log("Reading data from CSV ...")
    prefix = filename[0:3]
    df = pd.read_csv(filename, index_col="Unnamed: 0")
    features = df["strings"]
    labels = df["label"]
    from gensim.models.doc2vec import TaggedDocument
    import joblib

    bow = preprocessor.bagOfWords(df["strings"])[0]
    preDoc = preprocessor.doc2Vec([TaggedDocument(doc, [i]) for i, doc in enumerate(df["strings"])])
    doc2vec = [preDoc.infer_vector(x.split()) for x in df["strings"]]
    tfidf = preprocessor.TF_IDF(df["strings"])[0]

    print("BOW: ", bow.shape)
    print("DOC2VEC: ", len(doc2vec))
    print("TFIDF: ", tfidf.shape)
    
    t_bow, test, t_bow_l, lab = train_test_split(bow, labels, test_size=0.4, random_state=42, stratify=labels)
    v_bow, tt_bow, v_bow_l, tt_bow_l = train_test_split(test, lab, test_size=0.5, random_state=42, stratify=lab)

    t_doc2vec, test, t_doc2vec_l, lab = train_test_split(doc2vec, labels, test_size=0.4, random_state=42, stratify=labels)
    v_doc2vec, tt_doc2vec, v_doc2vec_l, tt_doc2vec_l = train_test_split(test, lab, test_size=0.5, random_state=42, stratify=lab)

    t_tfidf, test, t_tfidf_l, lab = train_test_split(tfidf, labels, test_size=0.4, random_state=42, stratify=labels)
    v_tfidf, tt_tfidf, v_tfidf_l, tt_tfidf_l = train_test_split(test, lab, test_size=0.5, random_state=42, stratify=lab)


    log.log("\n\nSaving Data ...\n\n")
    joblib.dump(t_bow, "DATA\\Train\\"+prefix+"bow_features.pkl")
    joblib.dump(t_bow_l, "DATA\\Train\\"+prefix+"bow_labels.pkl")
    joblib.dump(t_doc2vec, "DATA\\Train\\"+prefix+"doc2vec_features.pkl")
    joblib.dump(t_doc2vec_l, "DATA\\Train\\"+prefix+"doc2vec_labels.pkl")
    joblib.dump(t_tfidf, "DATA\\Train\\"+prefix+"tfidf_features.pkl")
    joblib.dump(t_tfidf_l, "DATA\\Train\\"+prefix+"tfidf_labels.pkl")

    joblib.dump(v_bow, "DATA\\Val\\"+prefix+"bow_features.pkl")
    joblib.dump(v_bow_l, "DATA\\Val\\"+prefix+"bow_labels.pkl")
    joblib.dump(v_doc2vec, "DATA\\Val\\"+prefix+"doc2vec_features.pkl")
    joblib.dump(v_doc2vec_l, "DATA\\Val\\"+prefix+"doc2vec_labels.pkl")
    joblib.dump(v_tfidf, "DATA\\Val\\"+prefix+"tfidf_features.pkl")
    joblib.dump(v_tfidf_l, "DATA\\Val\\"+prefix+"tfidf_labels.pkl")


    joblib.dump(tt_bow, "DATA\\Test\\"+prefix+"bow_features.pkl")
    joblib.dump(tt_bow_l, "DATA\\Test\\"+prefix+"bow_labels.pkl")
    joblib.dump(tt_doc2vec, "DATA\\Test\\"+prefix+"doc2vec_features.pkl")
    joblib.dump(tt_doc2vec_l, "DATA\\Test\\"+prefix+"doc2vec_labels.pkl")
    joblib.dump(tt_tfidf, "DATA\\Test\\"+prefix+"tfidf_features.pkl")
    joblib.dump(tt_tfidf_l, "DATA\\Test\\"+prefix+"tfidf_labels.pkl")

    log.log("Done splitting data!")
    # return (x_train, x_test, x_val), (y_train, y_test, y_val)


def stats(data, labels):
    d = {
        'len': data.shape[0],
        'features': data.shape[1],
        'count': labels.value_counts()
    }
    return d
