import json
import os


def createStats():
    benign_strings_compendium = []
    malicious_strings_compendium = []
    filenames = [file for file in os.listdir('Data\\Strings\\') if file.endswith('.txt')]
    benign_most_strings = 0
    benign_least_strings = float('inf')
    benign_total_num_strings = 0
    malicious_most_strings = 0
    malicious_least_strings = float('inf')
    malicious_total_num_strings = 0
    num_benign_files = 0
    num_malicious_files = 0
    for x in filenames:
        file = open("Data\\Strings\\" + x, 'r')
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
    benign_file_names = [f for f in os.listdir("Data\\Strings\\") if f.startswith('B_')]
    malicious_file_names = [f for f in os.listdir("Data\\Strings\\") if f.startswith('M_')]
    for x in benign_file_names:
        nested_dict = {}
        benign_strings[x] = nested_dict
        file = open("Data\\Strings\\" + x, 'r')
        all_lines = file.readlines()
        for line in all_lines:
            cleaned_line = line.replace('\n', '')
            if cleaned_line not in nested_dict:
                nested_dict[cleaned_line] = 1
            else:
                nested_dict[cleaned_line] += 1
        nested_dict['total'] = len(all_lines)
        file.close()
    for x in malicious_file_names:
        nested_dict = {}
        malicious_strings[x] = nested_dict
        file = open("Data\\Strings\\" + x, 'r')
        all_lines = file.readlines()
        for line in all_lines:
            cleaned_line = line.replace('\n', '')
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
                split_file = open(
                    "C:\\Users\\danee\\OneDrive\\Documents\\University\\Honours\\COS 700\\Year Project\\RDS\\Data\\Strings\\"
                    + x.replace('.json', '') + '.txt', 'w')
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
    filenames = os.listdir(
        "C:\\Users\\danee\\OneDrive\\Documents\\University\\Honours\\COS 700\\Year Project\\RDS\\Data\\Strings\\")
    for x in filenames:
        file = open(
            "C:\\Users\\danee\\OneDrive\\Documents\\University\\Honours\\COS 700\\Year Project\\RDS\Data\\Strings\\" + x, 'r')
        document = ''
        for y in file.readlines():
            document += y.replace('\n', '') + ' '
        corpus.append(document)
    return corpus
