import sys
import re
import nltk

import pandas as pd
import pickle
from sqlalchemy import create_engine

# import statements
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

import warnings
warnings.filterwarnings("ignore")

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    '''
    loads data from the database
    
    Input:
        database_filepath : filepath to the databse  
    Output:
        X: 'message' column 
        y: one-hot encoded categories
        categories_names: category names in y
    '''
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('MessageClassification', engine)
    
    
    # splitting the target
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)

    categories_names = Y.columns

    return X, Y, categories_names



def tokenize(text):
    ''' 
    Tokenizes the text into words, nomalizes it and performs lemmatization
    
    Input: 
        text: Raw Text
        
    Output:
        clean_tokens: Lemmatized tokens containing only alphanumeric characters 
    '''
      # normalize
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
    
      # Tokenize Sentence
    tokens = word_tokenize(text)
    
    # lematization
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(token, 'n').lower().strip() for token in tokens]
    clean_tokens = [lemmatizer.lemmatize(token, 'v').lower().strip() for token in tokens]
    
    return clean_tokens


def build_model():
    '''
    Returns the GridSearchCV object to be used as the model
    Input:
        None
    Output:
        cv (scikit-learn GridSearchCV): Grid search model object
    '''
    
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=tokenize)),  #create the CountVectorizer object
        ('tfidf', TfidfTransformer()),  #create Tfidftransformer object    
         ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1))#create the Classifier object
    ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function to evaluate the performance of model
 
    This function applies the model to test set and prints
    out the accuracy of the prediction of categories.
    
    '''
    '''
    Y_pred = model.predict(X_test)

    print("----Classification Report per Category:\n")
    for i in range(len(category_names)):
        print("Label:", category_names[i])
        print(classification_report(Y_test[:, i], Y_pred[:, i]))
        '''

    Y_pred = model.predict(X_test)
    

    Y_test=pd.DataFrame(data=Y_test,columns=category_names) #Convert prediction numpy into dataframe
    Y_pred=pd.DataFrame(data=Y_pred,columns=category_names)
    
    for column in Y_pred.columns:
        print(column)
        print(classification_report(Y_test[column], Y_pred[column]))
        print('_____________________________________________________')

    


def save_model(model, model_filepath):
    '''
    Save Model Function
    
    This Function saves trained model as Pickle file, to be loaded
    for prediction later. 
    '''

    model_pkl = open(model_filepath, 'wb')
    pickle.dump(model, model_pkl)



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