import pandas as pd
import numpy as np
import os


class Ensemble:

    def __init__(self, folder, filenames, output_folder, output_filename):
        self.folder = folder
        self.filenames = filenames
        self.output_filename = output_filename
        self.output_folder = output_folder
        self.dfs = []

    def read_df(self):
        for filename in filenames:
            self.dfs.append(pd.read_csv(os.path.join(self.folder, filename)))

    def average(self):
        print(self.dfs[0].shape)
        nb_samples = self.dfs[0].shape[0]
        res = np.zeros((nb_samples,))

        for df in self.dfs:
            res = np.add(res, df["Predicted"].values)

        res = res/len(self.dfs)
        self.res = res

    def save(self):
        ids = np.arange(self.res.shape[0])
        self.df_res = pd.DataFrame({'Id': range(self.res.shape[0]), 'Predicted': self.res})
        self.df_res.to_csv(os.path.join(self.output_folder, self.output_filename), index=False)

    def run(self):
        print("Reading CSV")
        self.read_df()
        print("Averaging")
        self.average()
        print("Saving")
        self.save()


if __name__ == "__main__":

    folder = "checkpoint"
    filenames = ["model_checkpoint_24_prediction.csv",
                 "model_checkpoint_27_prediction.csv",
                 "model_checkpoint_28_prediction.csv",
                 "cnn_rnn_model_prediction.csv",
                 "rnn_best_model_prediction.csv"]

    output_folder = "ensemble_output"
    output_filename = "ensemble_3CNN-28-27-24_2RNN.csv"

    ensemble = Ensemble(folder, filenames, output_folder, output_filename)

    ensemble.run()