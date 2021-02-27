import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import pickle

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import RegexpTokenizer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(database_filepath):
    """ load data and output features and targets
    """
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    file_name = database_filepath.split("/")[-1]
    
    table_name = file_name.split(".")[0]
    
    df = pd.read_sql_table(table_name, engine)
    
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = df.columns[4:]
    
    return X, Y, category_names



def tokenize(text):
    """ nature language process for the text
    """
    sentence = str(text)
    
    # Lowercase text
    sentence = sentence.lower()
    
    # Remove weblinks
    sentence = sentence.replace('{html}',"") 
    
    # Remove special characters
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url = re.sub(r'http\S+', '',cleantext)
    
    # Remove number
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num) 
    
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer() 
    
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    stem_words=[stemmer.stem(w) for w in filtered_words]
    lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]

    return lemma_words


def build_model():
    """ natural language process model
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        
        ])
    
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__n_estimators': [50, 100, 200]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
        
    """
    It takes too long to train the model. Therefore, I keep the parameter here.
    # set up the parameters
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    """
    return pipeline


def evaluate_model(model, X_test, y_test, category_names):
    y_pred = model.predict(X_test)
    
    # add the column name to the prediction
    y_pred = pd.DataFrame(y_pred, columns = category_names)
    
    for column in category_names:
        print(column)
        print(classification_report(y_test[column], y_pred[column]))
    

def save_model(model, model_filepath):
    # save the model to disk
    # https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/

    # 'wb' means allowing writing in binary
    pickle.dump(model, open(model_filepath, 'wb'))


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
