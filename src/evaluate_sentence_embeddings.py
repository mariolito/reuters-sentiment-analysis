import os
import sys
sys.path.append(os.path.join(".."))
from src.utils.store_utils import StorageHandler
store_handler = StorageHandler(dir_store=os.path.join(os.path.dirname(__file__), "..", "data"))
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s  - %(message)s', level=logging.INFO)
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def _read_data():
    X_train = store_handler.read('X_train_sentese_embds.p')
    Y_train = store_handler.read("df_train.p")["Y"].values
    X_test = store_handler.read('X_test_sentece_embds.p')
    Y_test = store_handler.read("df_test.p")["Y"].values
    return X_train, Y_train, X_test, Y_test


def fit_and_predict(X_train, Y_train, X_test, Y_test):
    clf = RandomForestClassifier(max_depth=100, n_estimators=1000)
    clf.fit(X_train, Y_train)
    Y_test_preds = clf.predict(X_test)
    print(80 * "*")
    print("OOS Evaluation Results")
    print("Confusion Matrix")
    conf_matrix = (confusion_matrix(Y_test, Y_test_preds))
    plt.figure(figsize=(9, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", linewidths=.5, xticklabels=np.unique(Y_test), yticklabels=np.unique(Y_test))
    plt.savefig(os.path.join("..", "data", 'confusion_matrix_sentence_embeddings.png'))
    print(conf_matrix)
    print("Matthews corrcoef")
    print(matthews_corrcoef(Y_test, Y_test_preds))
    print("Accuracy")
    print(accuracy_score(Y_test, Y_test_preds))


def run():
    X_train, Y_train, X_test, Y_test = _read_data()
    fit_and_predict(X_train, Y_train, X_test, Y_test)


if __name__ == "__main__":
    run()
