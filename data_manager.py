#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


class DataManager:

    def __init__(self, root_folder, train_path, target_path, test_path,
                 test_size=0.2, valid_size=0.2, sample_only=False,
                 rnn_model=False):
        self.root_folder = root_folder
        self.train_path = os.path.join(root_folder, train_path)
        self.target_path = os.path.join(root_folder, target_path)
        self.test_path = os.path.join(root_folder, test_path)
        self.test_size = test_size
        self.valid_size = valid_size
        self.rnn_model = rnn_model

        print("Train Path : %s" % self.train_path)
        print("Target Path : %s" % self.target_path)
        print("Test Path : %s" % self.test_path)

        self.load_data(sample_only)

    def load_data(self, sample_only):
        """
        data split into train, validation, test
        """

        X_path = os.path.join(self.train_path, '{0}.npy')
        # X_path = 'data/train/train/{0}.npy'
        y_path = self.target_path

        if sample_only:
            df = pd.read_csv(y_path, nrows=1000)
        else:
            df = pd.read_csv(y_path)
        y = df.Label.values

        y = np.zeros((df.shape[0],))
        X = np.zeros((df.shape[0], 1000, 102), dtype='float32')

        for id_label in df.itertuples():
            idx = id_label[1]
            label = id_label[2]

            data = np.load(X_path.format(idx))

            y[idx] = label
            X[idx, :data.shape[0], :] = data

        y = np_utils.to_categorical(y)

        if self.rnn_model:
            print("Reshaping the data for RNN model")
        else:
            print("Reshaping the data for CNN model")
            X = X.reshape(X.shape + (1,))

        if self.test_size == 0.0:
            X_train, y_train = X, y
            self.X_test = np.array([])
            self.y_test = np.array([])
        else:
            X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=self.test_size)

        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X_train, y_train, test_size=self.valid_size)

    def get_data(self):
        return (self.X_train, self.y_train), (self.X_valid, self.y_valid), (self.X_test, self.y_test)

    def load_predict(self):
        X_path = os.path.join(self.test_path, '{0}.npy')
        X = np.zeros((6051, 1000, 102), dtype='float32')

        for idx in range(6051):
            data = np.load(X_path.format(idx))
            X[idx, :data.shape[0], :] = data

        if self.rnn_model:
            self.X_pred = X
        else:
            self.X_pred = X.reshape(X.shape + (1,))

        return self.X_pred

    def convert_pred_to_submission(self, y_pred, filename):
        """
        construct csv file for upload
        """
        df = pd.DataFrame({'Id': range(6051), 'Predicted': y_pred[:, 1]})
        df.to_csv(filename, index=False)