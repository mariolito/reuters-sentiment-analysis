import tensorflow as tf
import pandas as pd
from collections import Counter
from sklearn.utils import shuffle
import os
import sys
sys.path.append(os.path.join(".."))
from src.utils.store_utils import StorageHandler
store_handler = StorageHandler(dir_store=os.path.join(os.path.dirname(__file__), "..", "data"))
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s  - %(message)s', level=logging.INFO)
import warnings
warnings.filterwarnings('ignore')


def _read_data():
    logging.info("Reading Data")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.reuters.load_data()
    word_index = tf.keras.datasets.reuters.get_word_index(
        path='reuters_word_index.json'
    )
    df_train = pd.DataFrame({"X": x_train, "Y": y_train})
    df_test = pd.DataFrame({"X": x_test, "Y": y_test})
    classes_counts = df_train["Y"].value_counts()
    classes = classes_counts[classes_counts > 400].index
    df_train = df_train[df_train["Y"].isin(classes)]
    df_test = df_test[df_test["Y"].isin(classes)]
    return df_train, df_test, word_index


def undersample(input_data, target):
    logging.info("Undersample majority class")
    min_occurence = min(Counter(list(input_data[target])).values())
    dfs = []
    sampled_indices = []
    for unique_target_value in input_data[target].unique():
        sampled_dataset = input_data[input_data[target] == unique_target_value].sample(min_occurence)
        sampled_indices += list(sampled_dataset.index)
        dfs.append(sampled_dataset)
    return shuffle(pd.concat(dfs))


def _store_data(df_train, df_test, word_index):
    logging.info("Store data")
    store_handler.store(df_train, 'df_train.p')
    store_handler.store(df_test, 'df_test.p')
    store_handler.store(word_index, 'word_index.p')


def run():

    df_train, df_test, word_index = _read_data()

    df_train = undersample(df_train, target="Y")

    _store_data(df_train, df_test, word_index)


if __name__ == "__main__":
    run()
