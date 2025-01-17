import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import sys


def load_data(messages_filepath, categories_filepath):
    
    '''
    Loading Messages and Categories from Destination Database
    Input: 
        messages_filepath = Path to the CSV containing file messages
        categories_filepath = Path to the CSV containing file categories    
    Output:
        df = Combined data containing messages and categories
    '''
    messages   = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merging them by id
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    ''' 
        Cleans the dataframe
         - renames columns of different categories
         - drops duplicates
         Args:
            df : Merged dataframe containing data from messages.csv and categories.csv
         Returns:
            df: Processed dataframe
    '''
    # split categories into separate category columns
    categories = df['categories'].str.split(pat = ';', expand = True)
    
    # select the first row of the categories and use it to  Renaming the column names
    row = categories.iloc[0,:].tolist()
    category_colnames = [col_name[:-2] for col_name in row ]
    categories.columns = category_colnames
    
    #replacing original values with 1 and 0
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = categories[column].astype(np.int)        
        
        
    # drop the original categories column from `df`
    
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    #remove duplicates
    df = df.drop_duplicates()
    df = df[df['related'].notna()]
    return df


def save_data(df, database_filename):
    '''
    Save Data to SQLite Database Function
    
    Input:
    - df: Clean Dataframe
    - database_filename: filename from user
    Ouput:
    Database saved in filepath name
    '''
   
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('MessageClassification', engine, index=False, if_exists='replace')  
  


def main():
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