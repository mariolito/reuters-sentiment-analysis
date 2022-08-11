import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.utils.tf_layer import set_input, add_layer
import tensorflow as tf
from tqdm import trange
import numpy as np
import math
from sklearn.metrics import accuracy_score
import warnings
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s  - %(message)s', level=logging.INFO)
warnings.filterwarnings('ignore')


class DL2DClassifier(object):

    def __init__(self, config):
        self.num_epochs = config['num_epochs']
        self.verbose = config['verbose']
        self.learning_rate = config['learning_rate']
        self.beta1 = config.get('beta1', 0.9)
        self.mini_batch_size = config['mini_batch_size']
        self.layers = config['layers']
        self.config = config

    def make_model(self):
        prev_name = ""
        model = tf.keras.Sequential()
        model = set_input(model, {'input_shape': self.input_shape})
        for config_layer in self.layers:
            model = add_layer(model, config_layer, prev_name)
            prev_name = config_layer['name']
        config_layer = {
            "name": "dense",
            "activation": self.config['activation_end'],
            "activation_config": self.config.get("activation_end_config", None),
            "batch_normalization": False,
            "dropout": False,
            "units": self.n_y
        }
        model = add_layer(model, config_layer, prev_name)

        return model

    def random_mini_batches(self, X, Y):
        mini_batches = []
        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(self.m))
        shuffled_X = X[permutation, :]
        shuffled_Y = Y[permutation, :]
        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        # number of mini batches of size mini_batch_size in your partitionning
        num_complete_minibatches = math.floor(self.m / self.mini_batch_size)
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[
                           k * self.mini_batch_size: k * self.mini_batch_size +
                                                     self.mini_batch_size, :]
            mini_batch_Y = shuffled_Y[
                           k * self.mini_batch_size: k * self.mini_batch_size +
                                                     self.mini_batch_size, :]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        # Handling the end case (last mini-batch < mini_batch_size)
        if self.m % self.mini_batch_size != 0:
            mini_batch_X = shuffled_X[
                           num_complete_minibatches * self.mini_batch_size:
                           self.m, :]
            mini_batch_Y = shuffled_Y[
                           num_complete_minibatches * self.mini_batch_size:
                           self.m, :]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        return mini_batches

    def train(self, X_train, Y_train, X_test=None, Y_test=None, validation=False):

        result = {}

        self.m = len(Y_train)

        self.input_shape = X_train.shape[1:]

        self.n_y = Y_train.shape[1]

        model = self.make_model()

        cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta1)

        loss_l = []
        t = trange(self.num_epochs, desc="Epoch back-propagation", leave=True)
        for epoch in t:
            minibatches = self.random_mini_batches(X_train, Y_train)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                with tf.GradientTape() as tape:

                    activations = model(minibatch_X)

                    loss = cross_entropy(minibatch_Y, activations)

                grads = tape.gradient(loss, model.trainable_variables)

                optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))

            if epoch % 5 == 0:
                loss_l.append(loss.numpy())

                msg = "Train acc: {} Test acc {}".format(
                    str(round(accuracy_score(tf.argmax(Y_train, 1).numpy(), tf.argmax(model(X_train).numpy(), 1).numpy()),
                              4)),
                    str(round(accuracy_score(tf.argmax(Y_test, 1).numpy(), tf.argmax(model(X_test).numpy(), 1).numpy()), 4))
                )
                t.set_description(msg)
                t.refresh()
            else:

                msg = "Loss: {}".format(str(round(loss.numpy(), 4)))
                t.set_description(msg)
                t.refresh()
        result.update({
            "model": model,
            "loss": loss_l
        })

        if validation:
            probs_train = model(X_train).numpy()
            probs_test = model(X_test).numpy()
            pred_train = tf.argmax(probs_train, 1).numpy()
            pred_test = tf.argmax(probs_test, 1).numpy()
            test_acc = accuracy_score(tf.argmax(Y_test, 1).numpy(), pred_test)
            train_acc = accuracy_score(tf.argmax(Y_train, 1).numpy(), pred_train)
            result.update({
                "Y_test": tf.argmax(Y_test, 1).numpy(),
                "pred_test": pred_test,
                "test_acc": test_acc,
                "probs_test": probs_test,
                "Y_train": tf.argmax(Y_train, 1).numpy(),
                "pred_train": pred_train,
                "train_acc": train_acc,
                "probs_train": probs_train
            })
            if self.verbose == 2:
                logging.info(
                    'Train completed.'
                    + ' train acc: '
                    + str(train_acc)[:4]
                    + ' test acc: '
                    + str(test_acc)[:4]
                )

        return result
