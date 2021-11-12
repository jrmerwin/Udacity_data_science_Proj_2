import sys


# import libraries
import sqlite3 
import pandas as pd 
from sqlalchemy import create_engine 
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer 
import nltk 
nltk.download(['punkt', 'wordnet']) 
import re 
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_multilabel_classification
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle

def load_data(database_filepath):
    '''
    Load dataframe from sqlite database and return features and target as X and Y
    Input: String filepath to the database (database_filepath) 
    Returns: X: Features and Y: Target 
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql('SELECT * FROM disaster_messages_db', con=engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    
    # Get the label names
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    '''
    tokenize, lemmatize, and clear out whitespace of passed messages
    input = one string from the message column 
    returns = array of tokenized message
    '''
    tokens = word_tokenize(text) 
    lemmatizer = WordNetLemmatizer() 
    clean_tokens = [] 
    for tok in tokens: 
        clean_tok = lemmatizer.lemmatize(tok).lower().strip() 
        clean_tokens.append(clean_tok) 
    return clean_tokens 


def build_model():
    '''
    Builds a multiOutputPlassifier using a pipeline and optimizes n_estimator parameters with Gridsearch 
    no inputs  
    output = Gridsearch with the pipeline and parameters passed to it    
    '''
    
    #define the pipeline
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    #define the parameters for gridsearch
    parameters = {
        'clf__estimator__n_estimators': [25, 50]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters) 
    return cv 


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Trains and evaluates the model, printing model performance metrics
    Inputs: the trained model, X and Y testing data, and the category names 
    Prints the results of the classification report      
    '''
    # Predict the given X_test and report the model performance metrics
    y_pred = model.predict(X_test)
    for index, column in enumerate(Y_test):
        print(column, classification_report(Y_test[column], y_pred[:, index]))


def save_model(model, model_filepath):
    """ 
    Exports the final model as a pickle file.
    """
    # Save the model based on model_filepath given
    filename_pkl = '{}'.format(model_filepath)
    with open(filename_pkl, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()