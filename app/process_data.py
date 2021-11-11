import sys
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    This function loads the two csv datasets and merges them.
    
    Input: filepaths to the two csv datasets 
    Output: a merged dataframe
    '''
    #load the datasets
    messages = pd.read_csv(messages_filepath) 
    categories = pd.read_csv(categories_filepath)
    
    #merge the datasets on key id
    df = pd.merge(messages, categories, how='inner', on = 'id') 
    
    #return the dataframe
    return df

def clean_data(df):
    '''
    This function cleans the data by converting the categories column into
    individual columns by splitting on the semi-colon delimeter, labels the 
    column names the row values, and converts each category value into a 
    numerical 0 or 1.
    
    Input: the merged dataframe from load_data()
    Output: a cleaned data dataframe ready to conversion to sql database
    '''
    # create a dataframe for the individual category columns 
    categories = df['categories'].str.split(";", expand=True) 

    # get category names from the first row of the categories dataframe 
    row = categories.loc[0]

    # make a list of column names for categories dataframe.
    category_colnames = row.apply(lambda x: x[:-2]).values.tolist() 
    
    # rename the columns using that list
    categories.columns = category_colnames
   

    # set each value to be the last character of the string using a for loop
    for column in categories:
        categories[column] = categories[column].str[-1:]  
    
        # now convert the column values to int 
        categories[column] = categories[column].astype(int) 
    
    #replace the categories column in merged df with the new columns
    df = df.drop("categories", 1) #where axis 1 = columns and axis 0 = rows 
    df = pd.concat([df, categories], axis=1) 

    #finally remove duplicate rows
    df = df.drop_duplicates()
    
    #return the cleaned dataframe
    return df

def save_data(df, database_filename):
    '''
    Builds a SQL database from the cleaned data
    
    Input: the cleaned dataframe and a name for the database file
    Output: a sqlite database
    '''
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('disaster_messages_db', engine, index=False)  


def main():
    ''' Runs the above functions to clean data and build sql database '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()