import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import model_selection, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix

import matplotlib.pyplot as plt



test_data = pd.read_csv("test.tsv", header=None, sep="\t", names=[
    'ID', 'Label', 'Statement', 'Subject', 'Speaker', 'Job', 'State', 'Party',
    'Barely True', 'False', 'Half True', 'Mostly True', 'Pants On Fire', 'Context'])

#test_data.head()

train_data = pd.read_csv("train.tsv", header=None, sep="\t", names=[
    'ID', 'Label', 'Statement', 'Subject', 'Speaker', 'Job', 'State', 'Party',
    'Barely True', 'False', 'Half True', 'Mostly True', 'Pants On Fire', 'Context'])
#train_data.head()

target_data = pd.read_csv("valid.tsv", header=None, sep="\t", names=[
    'ID', 'Label', 'Statement', 'Subject', 'Speaker', 'Job', 'State', 'Party',
    'Barely True', 'False', 'Half True', 'Mostly True', 'Pants On Fire', 'Context'])

#target_data.head()

train_data = pd.concat([train_data, test_data])
train_data = train_data.dropna()




def trainingAndPrediction(feature,print_graph):
    train_data_x, target_data_x, train_data_y, target_data_y = model_selection.train_test_split(train_data[feature],
                                                                                                train_data['Label'],
                                                                                                test_size=0.2)
    encoder = LabelEncoder()
    train_data_y = encoder.fit_transform(train_data_y)
    target_data_y = encoder.fit_transform(target_data_y)

    vectorize = TfidfVectorizer()
    vectorize.fit(train_data[feature])
    vector_train_data_x = vectorize.transform(train_data_x)
    vector_target_data_x = vectorize.transform(target_data_x)

    x = [vector_train_data_x, train_data_y]
    y = [vector_target_data_x, target_data_y]

    clf = MLPClassifier(solver='lbfgs',  hidden_layer_sizes=(10, ), random_state=1, max_iter=10000)
    clf.fit(vector_train_data_x, train_data_y)

    prediction = clf.predict(vector_target_data_x)

    print("MLP Classifier Accuracy Score of " ,feature," = ", accuracy_score(prediction, target_data_y) * 100)

    # if(print_graph):
    #     matrix = confusion_matrix(target_data_y, prediction,
    #                               labels=['barely-true', 'FALSE', 'TRUE', 'half-true', 'mostly-true', 'pants-fire'])
    #     print(matrix)
    #     plot_confusion_matrix(clf, vector_target_data_x, target_data_y)
    #     plt.show()

trainingAndPrediction('Statement',True)
trainingAndPrediction('Subject',False)
trainingAndPrediction('Speaker',False)
trainingAndPrediction('Job',False)
trainingAndPrediction('State',False)
trainingAndPrediction('Party',False)
trainingAndPrediction('Context',False)


