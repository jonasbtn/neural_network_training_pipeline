import json
import os
import pandas as pd
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.utils import multi_gpu_model
import h5py

from model import Model


class TrainingManager:

    def __init__(self, data_manager, training_config_path, checkpoint_filepath,
                 model_number=None, load_previous_model=False, previous_model_name=None,
                 multi_gpu=False, use_valid=True):

        self.data_manager = data_manager
        self.training_config_path = training_config_path
        self.checkpoint_filepath = checkpoint_filepath
        with open(self.training_config_path, 'r') as f:
            self.training_config = json.load(f)

        if not model_number:
            model_number = self.get_nb_model()
        self.model_number = model_number

        self.multi_gpu = multi_gpu

        if self.multi_gpu:
            self.training_config["model_checkpoint"]["activate"] = False

        self.use_valid = use_valid

        self.callbacks = self.set_callbacks()

        if self.multi_gpu:
            with tf.device("/cpu:0"):
                self.model = self.load_model(load_previous_model, previous_model_name)
            self.parallel_model = multi_gpu_model(self.model)
        else:
            self.model = self.load_model(load_previous_model, previous_model_name)

        self.compile_model()

    def set_callbacks(self):
        callbacks = []
        if self.training_config["callbacks"]:
            print("Callbacks activated")
            if "early_stopping" in self.training_config:
                es_config = self.training_config["early_stopping"]
                if es_config["activate"]:
                    es = EarlyStopping(monitor=es_config["monitor"],
                                       mode=es_config["mode"],
                                       patience=es_config["patience"],
                                       min_delta=es_config["min_delta"],
                                       verbose=es_config["verbose"])
                    callbacks.append(es)
                    print("Early Stopping Activated")
                else:
                    print("Early Stopping Deactivated")
                print("Early Stopping Config Loaded")
            if "model_checkpoint" in self.training_config:
                mc_config = self.training_config["model_checkpoint"]
                self.checkpoint_filename = mc_config["file_basename"].format(self.model_number)
                if mc_config["activate"]:
                    print("Saving model to %s" % self.checkpoint_filename)
                    mc = ModelCheckpoint(os.path.join(self.checkpoint_filepath, self.checkpoint_filename),
                                         monitor=mc_config["monitor"],
                                         mode=mc_config["mode"],
                                         save_best_only=mc_config["save_best_only"],
                                         verbose=mc_config["verbose"])
                    callbacks.append(mc)
                    print("Model CheckPoint Activated")
                else:
                    print("Model CheckPoint Deactivated")
                print("Model Checkpoint Config Loaded")
            if "reduce_lr_on_plateau" in self.training_config:
                reduce_config = self.training_config["reduce_lr_on_plateau"]
                if reduce_config["activate"]:
                    rlr = ReduceLROnPlateau(monitor=reduce_config["monitor"],
                                            mode=reduce_config["mode"],
                                            factor=reduce_config["factor"],
                                            patience=reduce_config["patience"],
                                            min_delta=reduce_config["min_delta"],
                                            cooldown=reduce_config["cooldown"],
                                            min_lr=reduce_config["min_lr"],
                                            verbose=reduce_config["verbose"])
                    callbacks.append(rlr)
                    print("Reduce Learning Rate on Plateau Activated")
        print("Callbacks Loaded")
        return callbacks

    def get_nb_model(self):
        files = os.listdir(self.checkpoint_filepath)
        nb_model = 0
        for file in files:
            if file[len(file)-2:len(file)] == "h5":
                nb_model += 1
        return nb_model

    def load_model(self, load_previous_model=False, previous_model_name=None):
        if load_previous_model:
            return load_model(os.path.join(self.checkpoint_filepath,previous_model_name))
        else:
            return Model()

    def save_model_json(self):
        print("Saving Model to Json")
        model_json = self.model.to_json()
        json_filename = '.'.join([self.checkpoint_filename.split('.')[0], 'json'])
        file_path = os.path.join(self.checkpoint_filepath, json_filename)
        with open(file_path, "w") as json_file:
            json_file.write(model_json)
        json_file.close()

    def compile_model(self):
        config = self.training_config["compile"]
        if self.multi_gpu:
            self.parallel_model.compile(loss=config["loss"],
                               optimizer=config["optimizer"],
                               metrics=config["metrics"])
        else:
            self.model.compile(loss=config["loss"],
                               optimizer=config["optimizer"],
                               metrics=config["metrics"])
        if config["verbose"]:
            print(self.model.summary())
            summary_filename = '.'.join([self.checkpoint_filename.split('.')[0]+"_summary", 'txt'])
            file_path = os.path.join(self.checkpoint_filepath, summary_filename)
            with open(file_path, 'w') as f:
                self.model.summary(print_fn=lambda x: f.write(x + '\n'))
            f.close()

    def run_training(self):
        (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = self.data_manager.get_data()
        if not self.use_valid:
            X_train = np.concatenate((X_train, X_valid))
            y_train = np.concatenate((y_train, y_valid))
            X_valid = X_test
            y_valid = y_test
            print("Not Using Valid set")
            print("Training on {} samples, Testing on {} samples".format(X_train.shape[0], X_test.shape[0]))
        config = self.training_config["fit"]
        print("Starting Training")
        if self.multi_gpu:
            self.history = self.parallel_model.fit(X_train, y_train,
                                                  validation_data=(X_valid, y_valid),
                                                  epochs=config["epochs"],
                                                  batch_size=config["batch_size"],
                                                  verbose=config["verbose"],
                                                  callbacks=self.callbacks)
            print("Training Over")
            if self.data_manager.test_size > 0.0:
                _, test_acc = self.parallel_model.evaluate(X_test, y_test)
        else:
            self.history = self.model.fit(X_train, y_train,
                                          validation_data=(X_valid, y_valid),
                                          epochs=config["epochs"],
                                          batch_size=config["batch_size"],
                                          verbose=config["verbose"],
                                          callbacks=self.callbacks)
            print("Training Over")
            if self.data_manager.test_size > 0.0:
                _, test_acc = self.model.evaluate(X_test, y_test)
        if self.data_manager.test_size > 0.0:
            accuracy_filename = '.'.join([self.checkpoint_filename.split('.')[0], 'txt'])
            file_path = os.path.join(self.checkpoint_filepath, accuracy_filename)
            with open(file_path, 'w') as f:
                f.write("Test Accuracy : {}".format(test_acc))
            f.close()
            print("Test Accuracy : %f" % test_acc)

    def save_history(self):
        # convert the history.history dict to a pandas DataFrame:
        hist_df = pd.DataFrame(self.history.history)
        csv_filename = '.'.join([self.checkpoint_filename.split('.')[0], 'csv'])
        file_path = os.path.join(self.checkpoint_filepath, csv_filename)
        print("Saving History to %s" % csv_filename)
        # save to csv:
        with open(file_path, mode='w') as f:
            hist_df.to_csv(f)
        f.close()
        print("History Saved")

    def load_best_model(self):
        file_path = os.path.join(self.checkpoint_filepath, self.checkpoint_filename)
        if os.path.isfile(file_path):
            model = load_model(file_path)
        else:
            model = self.model
        return model

    def predict(self):
        print("Predicting values from the best model")
        X_pred = self.data_manager.load_predict()
        best_model = self.load_best_model()
        y_pred = best_model.predict(X_pred)
        csv_filename = '.'.join([self.checkpoint_filename.split('.')[0]+"_prediction", 'csv'])
        file_path = os.path.join(self.checkpoint_filepath, csv_filename)
        self.data_manager.convert_pred_to_submission(y_pred, file_path)
        print("Prediction saved to : {}".format(file_path))

    def run(self):
        self.run_training()
        self.save_model_json()
        self.save_history()
        self.predict()