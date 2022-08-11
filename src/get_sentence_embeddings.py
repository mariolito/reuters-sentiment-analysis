import os
import sys
sys.path.append(os.path.join(".."))
from src.utils.get_embds_google_universal import Embeddor
from src.utils.store_utils import StorageHandler
store_handler = StorageHandler(dir_store=os.path.join(os.path.dirname(__file__), "..", "data"))
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s  - %(message)s', level=logging.INFO)
import warnings
warnings.filterwarnings('ignore')


def sentence_embeddings(list_of_words):
    logging.info("Extracting sentence embeddings")
    embds = Embeddor().get_embds_google(list_of_words)
    return embds


def _read_data():
    df_train = store_handler.read("df_train.p")
    df_test = store_handler.read("df_test.p")
    word_index = store_handler.read("word_index.p")
    return df_train, df_test, word_index


def _store_data(X_train_sentece_embds, X_test_sentece_embds):
    logging.info("Storing sentence embeddings")
    store_handler.store(X_train_sentece_embds, 'X_train_sentese_embds.p')
    store_handler.store(X_test_sentece_embds, 'X_test_sentece_embds.p')


def run():

    df_train, df_test, word_index = _read_data()

    index_word = {v: k for k, v in store_handler.read("word_index.p").items()}
    df_train["X_text"] = df_train.apply(lambda x: " ".join([index_word.get(i, "") for i in x["X"]]), axis=1)
    df_test["X_text"] = df_test.apply(lambda x: " ".join([index_word.get(i, "") for i in x["X"]]), axis=1)

    X_train_sentece_embds = sentence_embeddings(list(df_train["X_text"].values))
    X_test_sentece_embds = sentence_embeddings(list(df_test["X_text"].values))

    _store_data(X_train_sentece_embds, X_test_sentece_embds)


if __name__ == "__main__":
    run()
