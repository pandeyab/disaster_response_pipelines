#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[1]:


# import libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

import pandas as pd
import numpy as np
import pickle
import re

from sqlalchemy import create_engine

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score, make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD

import warnings
warnings.simplefilter('ignore')

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


nltk.download(['punkt', 'wordnet', 'stopwords'])


# In[3]:


# load data from database
engine = create_engine('sqlite:///DisasterResponse.db')
df = pd.read_sql("SELECT * FROM labeled_messages", engine)

X = df.filter(items=['id', 'message', 'original', 'genre'])
y = df.drop(['id', 'message', 'original', 'genre', 'child_alone'], axis=1)

#Mapping the '2' values in 'related' to '1' - because I consider them as a response (that is, '1')
y['related']=y['related'].map(lambda x: 1 if x == 2 else x)

df.head()


# In[4]:


display(type(X))
display(type(y))


# In[5]:


#just to see possible values distribution within classes
fig = plt.figure(figsize = (15,20))
ax = fig.gca()
y.hist(ax = ax)


# ### 2. Write a tokenization function to process your text data

# In[6]:


def tokenize(text):
    """Normalize, tokenize and stem text string
    
    parameters:
    text: string. String containing message
       
    Returns:
    stemmed: list of strings. List containing normalized and stemmed word tokens.
    """
    # Tokenizing words
    tokens = nltk.word_tokenize(text)
    # Stem word tokens and remove stop words
    lemmatizer = nltk.WordNetLemmatizer()
    
    stemmed = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
    
    return stemmed


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# 
# pipeline = Pipeline([
#                     ('vect', CountVectorizer(tokenizer=tokenize)),
#                     ('tfidf', TfidfTransformer()),
#                     ('clf', MultiOutputClassifier(RandomForestClassifier()))
#                     ])

# In[14]:


pipeline = Pipeline([('cvect', CountVectorizer(tokenizer = tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', RandomForestClassifier())
                     ])


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[8]:


#Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)


# In[9]:


display(np.unique(y_train))
display(np.unique(y_test))


# In[10]:


#train pipeline
np.random.seed(15)
pipeline.fit(X_train['message'], y_train)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[11]:


#prediction with training set and test set
y_pred_test = pipeline.predict(X_test['message'])
y_pred_train = pipeline.predict(X_train['message'])

print(classification_report(y_test.values, y_pred_test, target_names=y.columns.values))
print()
print('\n',classification_report(y_train.values, y_pred_train, target_names=y.columns.values))


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[15]:


parameters = {'clf__max_depth': [10, 20, None],
              'clf__min_samples_leaf': [1, 2, 4],
              'clf__min_samples_split': [2, 5, 8],
              'clf__n_estimators': [20, 50]}

cv = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_micro', verbose = 18, n_jobs=-1)


# In[16]:


improved_model = cv.fit(X_train['message'], y_train)


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[17]:


y_pred_test_1 = improved_model.predict(X_test['message'])
y_pred_train_1 = improved_model.predict(X_train['message'])

print(classification_report(y_test.values, y_pred_test_1, target_names=y.columns.values))
print()
print('\n',classification_report(y_train.values, y_pred_train_1, target_names=y.columns.values))


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[18]:


# Using Decision Tree Classifier now 
pipeline_new = Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultiOutputClassifier(
                        AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, class_weight='balanced'))
                    ))
                        ])

# Improved parameters 
parameters_new = {
                'clf__estimator__n_estimators': [100, 200],
                'clf__estimator__learning_rate': [0.1, 0.3]
                }


# In[20]:


# new model with improved parameters
cv2 = GridSearchCV(pipeline_new, param_grid=parameters_new, cv=3, scoring='f1_micro', verbose=10)


# In[ ]:


#further improved model
improved_model_2 = cv2.fit(X_train['message'], y_train)


# In[ ]:


improved_model_2.best_params_


# In[ ]:


# prediction with test set and training set
y_pred_test_2 = improved_model_2.predict(X_test['message'])
y_pred_train_2 = improved_model_2.predict(X_train['message'])

print(classification_report(y_test.values, y_pred_test_2, target_names=y.columns.values))
print()
print('\n',classification_report(y_train.values, y_pred_train_2, target_names=y.columns.values))


# ### 9. Export your model as a pickle file

# In[ ]:


# Pickle best model
pickle.dump(improved_model_2, open('final_disaster_model.sav', 'wb'))


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




