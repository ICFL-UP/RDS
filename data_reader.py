import log

import json
import os
import random
from sklearn.model_selection import train_test_split
import pandas as pd


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
    
    df = pd.read_csv(filename, index_col="Unnamed: 0")
    features = df["strings"]
    labels = df["label"]
    
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=42, stratify=labels)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42, stratify=y_test)
    log.log("Training: " + str(round(len(y_train) / len(labels), 2)))
    log.log("Validation: " + str(round(len(y_val) / len(labels), 2)))
    log.log("Testing: " + str(round(len(y_test) / len(labels), 2)))
    
    log.log("Done splitting data!")
    return (x_train, x_test, x_val), (y_train, y_test, y_val)