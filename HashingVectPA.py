import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split

trainingD = pd.read_csv("train.tsv", header=None, sep="\t", names=[
    'ID', 'Label', 'Statement', 'Subject', 'Speaker', 'Job', 'State', 'Party',
    'Barely True', 'False', 'Half True', 'Mostly True', 'Pants On Fire', 'Context'])

testingD = pd.read_csv("test.tsv", header=None, sep="\t", names=[
    'ID', 'Label', 'Statement', 'Subject', 'Speaker', 'Job', 'State', 'Party',
    'Barely True', 'False', 'Half True', 'Mostly True', 'Pants On Fire', 'Context'])

df = pd.concat([trainingD, testingD])
df = df.dropna()
passive_aggressive = PassiveAggressiveClassifier(max_iter=500)

x_train, x_test, y_train, y_test = train_test_split(df["Statement"], df['Label'], test_size=0.2)


hashvec = HashingVectorizer(stop_words='english', n_features=15)
hashvec_train = hashvec.fit_transform(x_train)
hashvec_test = hashvec.fit_transform(x_test)

passive_aggressive.fit(hashvec_train, y_train)

y_pred = passive_aggressive = passive_aggressive.predict(hashvec_test)
score = accuracy_score(y_test, y_pred)
print(f' Hash Vector Accuracy: {round(score * 100, 2)}%')
