import os
import sys
sys.path.append(os.path.join(".."))
from src.utils.store_utils import StorageHandler
store_handler = StorageHandler(dir_store=os.path.join(os.path.dirname(__file__), "..", "data"))
import logging
import pandas as pd
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s  - %(message)s', level=logging.INFO)
import warnings
warnings.filterwarnings('ignore')
from src.utils.dl_2D_classification import DL2DClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


max_sent_len = 500
cnn_config = {
    "num_epochs": 10,
    "verbose": 2,
    "mini_batch_size": 32,
    "learning_rate": 0.0005,
    "layers": [
        {"name": "embedding", "input_dim": 0, "output_dim": 400, "input_length": max_sent_len},
        {"name": "cnn1d", "filters": 32, "kernel_size": 3, "strides": 1, "padding": "same", "activation": "leaky_relu", "activation_config": {"alpha": 0.2}, "kernel_initializer": "he_uniform"},
        {"name": "maxpool1D", "pool_size": 3, "strides": 3, "padding": "valid", "dropout": 0.70}
    ],
    "activation_end": "softmax",
    "activation_end_config": {},
    "beta1": 0.5
}


def _read_data():
    df_train = store_handler.read("df_train.p")
    df_test = store_handler.read("df_test.p")
    word_index = store_handler.read("word_index.p")
    return df_train, df_test, word_index


def _preprocess(df_train, df_test, word_index):

    X_train = pad_sequences(df_train["X"].values, padding='post', value=0, maxlen=max_sent_len)
    X_test = pad_sequences(df_test["X"].values, padding='post', value=0, maxlen=max_sent_len)

    index_to_word = {0: "<PAD>", 1: "<s>", 2: "<UNK>"}
    for key, value in word_index.items():
        index_to_word[value + 3] = key

    tmp_dummies = pd.get_dummies(df_train[['Y']].astype(str))
    constructed_features = list(tmp_dummies)
    df_train = pd.concat([df_train, tmp_dummies], axis=1)
    Y_train = df_train[constructed_features].values

    tmp_dummies = pd.get_dummies(df_test[['Y']].astype(str))
    constructed_features = list(tmp_dummies)
    df_test = pd.concat([df_test, tmp_dummies], axis=1)
    Y_test = df_test[constructed_features].values

    Y_unique = df_test["Y"].astype(str).sort_values().unique()

    return X_train, Y_train, X_test, Y_test, index_to_word, Y_unique


def fit_and_predict(X_train, Y_train, X_test, Y_test, index_to_word, Y_unique):
    cnn_config['layers'][0]['input_dim'] = len(index_to_word)
    clf = DL2DClassifier(config=cnn_config)
    result = clf.train(X_train, Y_train, X_test, Y_test, validation=True)
    print(80 * "*")
    print("OOS Evaluation Results")
    print("Confusion Matrix")
    conf_matrix = (confusion_matrix(result['Y_test'], result['pred_test']))
    plt.figure(figsize=(9, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", linewidths=.5, xticklabels=Y_unique,
                yticklabels=Y_unique)

    plt.savefig(os.path.join("..", "data", 'confusion_matrix_sequential_embeddings.png'))

    print(conf_matrix)
    print("Matthews corrcoef")
    print(matthews_corrcoef(result['Y_test'], result['pred_test']))
    print("Accuracy")
    print(accuracy_score(result['Y_test'], result['pred_test']))




def main():
    df_train, df_test, word_index = _read_data()
    X_train, Y_train, X_test, Y_test, index_to_word, Y_unique = _preprocess(df_train, df_test, word_index)
    fit_and_predict(X_train, Y_train, X_test, Y_test, index_to_word, Y_unique)


if __name__ == "__main__":
    main()
