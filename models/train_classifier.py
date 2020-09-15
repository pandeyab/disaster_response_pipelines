# import libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

import pandas as pd
import numpy as np
import pickle
import re
import sys

from sqlalchemy import create_engine

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, fbeta_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt


nltk.download(['punkt', 'wordnet', 'stopwords'])

#define function for loading data
def data_loading(database_filepath: str):
    '''
    loading database
    
    parameters: the path of the database
    Returns:    X: features (messages)
                y: categories
                An ordered list of categories
    '''
    # Loading the database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM labeled_messages",engine) 
    print(df.head(5))

    # Making X DataFrame with only message feature
    X = df['message']

    # Making Y DataFrame with required categories
    y = df.drop(['id', 'message', 'original', 'genre'],  axis=1).astype(float)

    #define Y columns as list
    categories = list(y.columns.values)
    print(categories)

    return X, y, categories


#define function for tokenize the strings
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


#define function for model building
def model_building():
    '''
    pipeline builder function
    parameters: None
    Returns: Model
    '''
    # Pipeline
    pipeline = Pipeline([('cvect', CountVectorizer(tokenizer = tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', RandomForestClassifier())
                     ])
    
    # Parameters
    parameters = {'clf__min_samples_leaf': [1, 2, 4],
                 'clf__min_samples_split': [5, 10, 15],
                  'clf__n_estimators': [20, 50]
                 }

    # GridSearch application
    cv_imp  = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_micro', verbose = 30, n_jobs=-1)

    return cv_imp


#define function for model evaluation
def model_eval(model, X_test, Y_test, category_names):
    '''
    Model evaluation function which gives accuracy, f1_score, recall_score and precision_score
    parameters:   Model, test dataframes, categories list
    Returns: Classification report
    '''
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))

    for  idx, cat in enumerate(Y_test.columns.values):
        print("category : {} -- accuracy_score : {} -- recall_score : {} -- precision_score: {} -- f1_score : {}".format(cat,
                                                accuracy_score(Y_test.values[:,idx], y_pred[:, idx]),
                                                recall_score(Y_test.values[:,idx], y_pred[:,idx], average='weighted'),
                                                precision_score(Y_test.values[:,idx], y_pred[:,idx], average='weighted'),
                                                f1_score(Y_test.values[:,idx], y_pred[:,idx], average='weighted')))
    print("accuracy = {}".format(accuracy_score(Y_test, y_pred)))
    print("recall_score = {}".format(recall_score(Y_test, y_pred, average='weighted')))
    print("precision_score = {}".format(precision_score(Y_test, y_pred, average='weighted')))
    print("f1_score = {}".format(f1_score(Y_test, y_pred, average='weighted')))


#define function for saving the model
def model_save(model, model_filepath):
    '''
    Model saving function as pickle file
    parameters: Model, filepath
    Returns: pikcle file
    '''
    with open(model_filepath, 'wb') as file:  
        pickle.dump(model, file)


#main function
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = data_loading(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = model_building()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        model_eval(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        model_save(model, model_filepath)

        print('saving trained model..')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


#calling main function
if __name__ == '__main__':
    main()