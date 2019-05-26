#!/usr/bin/env python
# coding: utf-8


# Necessary imports
import nltk
import glob
import scipy
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split



pos_files = glob.glob("data/clean_positive_train_*.csv")
neg_files = glob.glob("data/clean_negative_train_*.csv")

df_pos_list = [pd.read_csv(open(fp, 'r'), encoding='utf-8', engine='c') for fp in pos_files[:10]]
df_neg_list = [pd.read_csv(open(fp, 'r'), encoding='utf-8', engine='c') for fp in neg_files[:10]]


# Now we make features for each dataset that we have, calculating all these features takes a long long time...



df = [pd.concat([df_pos, df_neg]) for (df_pos, df_neg) in zip(df_pos_list, df_neg_list)]
print(len(df))

# Now we have scrambeled dataframes of 20000 entries, with features such as binary scores and profanity
df[0].describe()


# In[104]:


df[0].head()


# In[105]:


tokenizer = Tokenizer(num_words=10000, lower=True, split=' ', document_count=0)
# Create the word_index list based on all our data",
text_data = [np.array2string(df[single_df]['text'].values.astype(str)) for single_df in range(len(df))]
text_data = ' '.join(text_data)
tokenizer.fit_on_texts(text_data)


# In[110]:


feature_data = [np.array([])]
score_data = [np.array([])]

X_train = []
X_test = []
y_train = []
y_test = []

# Iterate over each DataFrame
for n in range(len(df)):
    print(n)
    cur_df = df[n]
    df[n].dropna(axis=0, inplace=True)
    size = len(cur_df)
    # Iterate over each row in the DataFrame
    for index, row in cur_df.iterrows():
        sentiment = row['sentiment']
        profanity = row['profanity_prob']
        features = np.hstack((sentiment, profanity))
        if index == 0:
            feature_data[n] = features
            score_data[n] = row['score']
        else:
            feature_data[n] = np.vstack([feature_data[n], features])
            score_data[n] = np.vstack([score_data[n], row['score']])
            
    X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = train_test_split(feature_data[n], score_data[n], test_size=0.20, random_state=42)
    X_train.append(X_train_tmp)
    X_test.append(X_test_tmp)
    y_train.append(y_train_tmp)
    y_test.append(y_test_tmp)

    X_train[n] = tf.keras.utils.normalize(X_train[n], axis=1)
    X_test[n] = tf.keras.utils.normalize(X_test[n], axis=1)

print(X_train[0])
print(y_train[0])


# In[ ]:





# ## Finally we can do some neural networks!

# In[91]:


model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(24, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

# model.summary()


# In[92]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])


# In[94]:


test_number = 1

train_data = X_train[test_number]
test_data = X_test[test_number]

history = model.fit(train_data,
                    y_train[test_number],
                    epochs=40,
                    validation_data=(test_data, y_test[test_number]),
                    verbose=1)


# In[ ]:




