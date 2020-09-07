# Disaster Response Pipelines
Udacity DSND project 5 - createing a pieline for Disaster Response app with help of ETL, NLP and ML 

### Table of Contents

1. [Installation](#installation)
2. [Project Overview](#overview)
3. [Data Required for project](#data)
4. [File Description](#files)
5. [Commands for execution](#commands)
6. [Licensing, Authors, and Acknowledgements](#licensing)


## Installation <a name="installation"></a>

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- JSON (https://www.json.org/json-en.html)
- nltk (https://www.nltk.org/)
- sklearn (https://scikit-learn.org/stable/)
- flask (https://flask.palletsprojects.com/en/1.1.x/)
- sqlalchemy (https://www.sqlalchemy.org/)
- re
- pickle 
- warnings (https://docs.python.org/3/library/warnings.html)
- [matplotlib](http://matplotlib.org/)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

We recommend students install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project. 

## Project Overview <a name="overview"></a>
This repository contains code for a web app which an emergency worker could use during a disaster event (e.g. an earthquake or hurricane), to classify a disaster message into several categories, in order that the message can be directed to the appropriate aid agencies. 

The app uses a ML model to categorize any new messages received, and the repository also contains the code used to train the model and to prepare any new datasets for model training purposes.

## Data Required for poject <a name="data"></a>
* **data**: This folder contains sample messages and categories datasets in csv format.

## Files Description <a name="files"></a>
* **process_data.py**: This code takes as its input csv files containing message data and message categories (labels), and creates an SQLite database containing a merged and cleaned version of this data.
* **train_classifier.py**: This code takes the SQLite database produced by process_data.py as an input and uses the data contained within it to train and tune a ML model for categorizing messages. The output is a pickle file containing the fitted model. Test evaluation metrics are also printed as part of the training process.
* **ETL Pipeline Preparation.ipynb**: The code and analysis contained in this Jupyter notebook was used in the development of process_data.py. process_data.py effectively automates this notebook.
* **ML Pipeline Preparation.ipynb**: The code and analysis contained in this Jupyter notebook was used in the development of train_classifier.py. In particular, it contains the analysis used to tune the ML model and determine which algorithm to use. train_classifier.py effectively automates the model fitting process contained in this notebook.


## Commands for execution <a name="commands"></a>
### ***Run process_data.py***
1. Save the data folder in the current working directory and process_data.py in the data folder.
2. From the current working directory, run the following command:
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

### ***Run train_classifier.py***
1. In the current working directory, create a folder called 'models' and save train_classifier.py in this.
2. From the current working directory, run the following command:
`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

### ***Run the web app***
1. Save the app folder in the current working directory.
2. Run the following command in the app directory:
    `python run.py`
3. Go to http://0.0.0.0:3001/


## Licensing, Authors, Acknowledgements
This app was completed as part of the [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025). Code templates and data were provided by Udacity. The data was originally sourced by Udacity from Figure Eight.
