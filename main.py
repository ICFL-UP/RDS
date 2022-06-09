from sklearn.feature_extraction.text import TfidfVectorizer
import data_reader
import os


def main():
    # data_reader.createStats()

    # Trying TF_IDF
    corpus = []
    filenames = os.listdir('Data/Strings')
    for x in filenames:
        file = open("Data/Strings/" + x, 'r')
        document = ''
        for y in file.readlines():
            document += y.replace('\n', '') + ' '
        corpus.append(document)
    vectorizer = TfidfVectorizer(token_pattern=r"\S{2,}")
    X = vectorizer.fit_transform(corpus)
    print(X)


if __name__ == "__main__":
    main()
