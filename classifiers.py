from sklearn.ensemble import RandomForestClassifier


def randomForrest(train_data, correct_class):
    classifier = RandomForestClassifier()
    classifier.fit(train_data, correct_class)
    return classifier
