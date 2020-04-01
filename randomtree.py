import re

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix


def process(features):
    processed_features = []
    for sentence in range(0, len(features)):
        # Remove all the special characters
        processed_feature = re.sub(r'\W', ' ', str(features[sentence]))

        # remove all single characters
        processed_feature = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

        # Remove single characters from the start
        processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)

        # Substituting multiple spaces with single space
        processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

        # Removing prefixed 'b'
        processed_feature = re.sub(r'^b\s+', '', processed_feature)

        # Converting to Lowercase
        processed_feature = processed_feature.lower()

        processed_features.append(processed_feature)
    return processed_features


test = pd.read_csv("test.tsv", header=None, sep="\t", names=[
    'ID', 'Label', 'Statement', 'Subject', 'Speaker', 'Job', 'State', 'Party',
    'Barely True', 'False', 'Half True', 'Mostly True', 'Pants On Fire', 'Context'])

train = pd.read_csv("train.tsv", header=None, sep="\t", names=[
    'ID', 'Label', 'Statement', 'Subject', 'Speaker', 'Job', 'State', 'Party',
    'Barely True', 'False', 'Half True', 'Mostly True', 'Pants On Fire', 'Context'])

valid = pd.read_csv("valid.tsv", header=None, sep="\t", names=[
    'ID', 'Label', 'Statement', 'Subject', 'Speaker', 'Job', 'State', 'Party',
    'Barely True', 'False', 'Half True', 'Mostly True', 'Pants On Fire', 'Context'])

features = pd.concat([test, train])
features.pop("ID")
features.pop("Barely True")
features.pop("False")
features.pop("Half True")
features.pop("Mostly True")
features.pop("Pants On Fire")
features.pop("Context")
one = features.pop("Statement")
labels = np.array(features.pop("Label"))
print(features.shape)
print(labels.shape)
class_names = test.Label
vectorizer = TfidfVectorizer(max_features=7600, stop_words=stopwords.words('english'))
processed_features = vectorizer.fit_transform(one)

X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, stratify=labels, test_size=0.25,
                                                    random_state=123)
#
text_classifier = RandomForestClassifier(n_estimators=700, verbose=1,
                                         n_jobs=-1)
text_classifier.fit(X_train, y_train)

# pipeline stuff, was used to tune parameters

# pipe = Pipeline([('tf', TfidfVectorizer()),
#                ('rf', RandomForestClassifier())])

# Tune GridSearchCV
# pipe_params = {'tf__ngram_range': [(1, 10), (5, 20)],
#               'tf__max_features': [7600],
#               'rf__n_estimators': [500, 600, 700],
#               'rf__n_jobs': [-1]}

# gs = GridSearchCV(pipe, param_grid=pipe_params, cv=3)
# gs.fit(X_train, y_train)
# print("Best score:", gs.best_score_)
# print("Train score", gs.score(X_train, y_train))
# print("Test score", gs.score(X_test, y_test))
# print(gs.best_params_)


disp = plot_confusion_matrix(text_classifier, X_test, y_test,
                             display_labels=class_names,
                             cmap=plt.cm.Blues)
disp.ax_.set_title("Confusion matrix, without normalization")

print("Confusion matrix of RandomForestClassifier")
print(disp.confusion_matrix)

plt.show()
predictions = text_classifier.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))
