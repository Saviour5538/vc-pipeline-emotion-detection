import numpy as np
import pandas as pd

import os

from sklearn.feature_extraction.text import CountVectorizer

# fetch the data from data/processed
train_data = pd.read_csv('./data/processed/train_processed.csv')
test_data = pd.read_csv('./data/processed/test_processed.csv')
 
# Fill NaN values with empty strings
X_train = train_data['content'].fillna("").values
y_train = train_data['sentiment'].values

X_test = test_data['content'].fillna("").values
y_test = test_data['sentiment'].values


# apply bag of words (countvectorizer)
vectorizer = CountVectorizer()

# fit the vectorizer on the training data and transform it
X_train_bow = vectorizer.fit_transform(X_train)

# transform the test data using the same vectorizer
X_test_bow = vectorizer.transform(X_test)

train_df = pd.DataFrame(X_train_bow.toarray())
train_df['label'] = y_train

test_df = pd.DataFrame(X_test_bow.toarray())
test_df['label'] = y_test

# store the data inside data/features

data_path = os.path.join("data","features")

os.makedirs(data_path,exist_ok=True)

train_df.to_csv(os.path.join(data_path,"train_bow.csv"),index=False)
test_df.to_csv(os.path.join(data_path,"test_bow.csv"),index=False) 