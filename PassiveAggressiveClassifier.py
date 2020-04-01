import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2 as chi
from sklearn.feature_selection import SelectKBest as Kbest

trainingD = pd.read_csv("train.tsv", header=None, sep="\t", names=[
    'ID', 'Label', 'Statement', 'Subject', 'Speaker', 'Job', 'State', 'Party',
    'Barely True', 'False', 'Half True', 'Mostly True', 'Pants On Fire', 'Context'])

testingD = pd.read_csv("test.tsv", header=None, sep="\t", names=[
    'ID', 'Label', 'Statement', 'Subject', 'Speaker', 'Job', 'State', 'Party',
    'Barely True', 'False', 'Half True', 'Mostly True', 'Pants On Fire', 'Context'])

df = pd.concat([trainingD, testingD])
df = df.dropna()

passive_aggressive = PassiveAggressiveClassifier(max_iter=500)
tfidf = TfidfVectorizer(stop_words='english', max_df=0.9)

x_train, x_test, y_train, y_test = train_test_split(df["Party"], df['Label'], test_size=0.2)

tfidf_train = tfidf.fit_transform(x_train)
tfidf_test = tfidf.transform(x_test)

passive_aggressive.fit(tfidf_train, y_train)

y_pred = passive_aggressive.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f' passive aggressive Accuracy: {round(score * 100, 2)}%')

matrix = confusion_matrix(y_test, y_pred,
                          labels=['barely-true', 'FALSE', 'TRUE', 'half-true', 'mostly-true', 'pants-fire'])
print(matrix)
plot_confusion_matrix(passive_aggressive, tfidf_test, y_test)
plt.show()
