# import packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# import data
spam_df = pd.read_csv("./spam.csv")

# inspect data
print(spam_df.groupby('Category').describe())

# turn spam/ham into numerical data, creating a new column called spam
spam_df['spam'] = spam_df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
print(spam_df['spam'].head())

# create train/test split
x_train, x_test, y_train, y_test = train_test_split(spam_df.Message, spam_df.spam, test_size=0.25)
print(len(x_train), len(x_test))

# find word count and store data as a matrix
cv = CountVectorizer()
x_train_count = cv.fit_transform(x_train.values)
print(x_train_count.shape)

# train model
model = MultinomialNB()
model.fit(x_train_count, y_train)

# pre-test ham
email_ham = ["cricket tickets later"]
email_ham_count = cv.transform(email_ham)
print(model.predict(email_ham_count))

# pre-test spam
email_spam = ["reward money click"]
email_spam_count = cv.transform(email_spam)
print(model.predict(email_spam_count))

# test model
x_test_count = cv.transform(x_test)
print(model.score(x_test_count, y_test))
