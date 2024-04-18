import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

spam_df = pd.read_csv("./spam.csv")

print(spam_df.groupby('Category').describe())

spam_df['spam'] = spam_df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
print(spam_df['spam'].head())

x_train, x_test, y_train, y_test = train_test_split(spam_df.Message, spam_df.spam, test_size=0.25)
print(len(x_train), len(x_test))

cv = CountVectorizer()
x_train_count = cv.fit_transform(x_train.values)
print(x_train_count.shape)

model = MultinomialNB()
model.fit(x_train_count, y_train)

email_ham = ["cricket tickets later"]
email_ham_count = cv.transform(email_ham)
print(model.predict(email_ham_count))

email_spam = ["reward money click"]
email_spam_count = cv.transform(email_spam)
print(model.predict(email_spam_count))

x_test_count = cv.transform(x_test)
print(model.score(x_test_count, y_test))
