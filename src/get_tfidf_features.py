from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
import os
import sys
sys.path.append(os.path.join(".."))
from src.utils.store_utils import StorageHandler
store_handler = StorageHandler(dir_store=os.path.join(os.path.dirname(__file__), "..", "data"))
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s  - %(message)s', level=logging.INFO)
import warnings
warnings.filterwarnings('ignore')


min_df = 3
ngram_range = (1, 4)
token_pattern = "\S+"
num_features = 200
index_word = {v: k for k, v in store_handler.read("word_index.p").items()}


def _read_data():
    df_train = store_handler.read("df_train.p")
    df_test = store_handler.read("df_test.p")
    return df_train, df_test

def extract_features(df_train, df_test):
    x_train = df_train['X'].values
    x_test = df_test['X'].values

    x_train_astext = [" ".join([str(i) for i in x_train[j]]) for j in range(len(x_train))]
    x_test_astext = [" ".join([str(i) for i in x_test[j]]) for j in range(len(x_test))]

    vect = TfidfVectorizer(min_df=min_df, ngram_range=ngram_range,
                                            token_pattern=token_pattern).fit(x_train_astext)
    X_train_vect = vect.transform(x_train_astext)
    X_test_vect = vect.transform(x_test_astext)
    all_features = np.array(vect.get_feature_names())

    kb = SelectKBest(f_classif, k=200)

    X_train_vect = kb.fit_transform(X_train_vect, df_train['Y'].values)
    X_test_vect = kb.transform(X_test_vect)



    selection_indices = np.array([i for i in kb.get_support(indices=True)])
    model_importances = dict(zip(all_features[selection_indices], kb.scores_[selection_indices]))

    return X_train_vect, X_test_vect, model_importances


def _store_features(X_train_vect, X_test_vect, model_importances):
    store_handler.store(X_train_vect, 'X_train_vect.p')
    store_handler.store(X_test_vect, 'X_test_vect.p')
    store_handler.store(model_importances, 'model_importances.p')


def run():
    df_train, df_test = _read_data()
    logging.info("Extract TFIDF features")
    X_train_vect, X_test_vect, model_importances = extract_features(df_train, df_test)
    logging.info("Store TFIDF features")
    _store_features(X_train_vect, X_test_vect, model_importances)


if __name__ == "__main__":
    run()
