import argparse
import json

from data_manager import DataManager
from training_manager import TrainingManager


class RunTraining:

    def __init__(self, training_config_path, model_number=None,
                 sample_only=False, multi_gpu=False, use_valid=True, rnn_model=False):

        with open(training_config_path, 'r') as f:
            self.training_config = json.load(f)

        print("Loading files")
        path = self.training_config["path"]
        root_folder = path["root_folder"]
        train_path = path["train_path"]
        target_path = path["target_path"]
        test_path = path["test_path"]
        test_size = self.training_config["test_size"]
        valid_size = self.training_config["valid_size"]
        self.data_manager = DataManager(root_folder, train_path, target_path, test_path, test_size=test_size,
                                        valid_size=valid_size, sample_only=sample_only, rnn_model=rnn_model)
        print("Files Loaded")

        print("Loading Training Configuration")
        checkpoint_filepath = path["checkpoint_filepath"]
        load_config = self.training_config["load"]
        load_previous_model = load_config["load_previous_model"]
        previous_model_name = load_config["previous_model_name"]
        self.training_manager = TrainingManager(self.data_manager, training_config_path, checkpoint_filepath,
                                                model_number, load_previous_model, previous_model_name,
                                                multi_gpu, use_valid)
        print("Training Configuration Loaded")

    def run(self):
        self.training_manager.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', dest='training_config_path', default='training_config.json')
    parser.add_argument('-n', dest='model_number', default=None)
    parser.add_argument('-s', dest='sample_only', default=False)
    parser.add_argument('-t', dest='use_valid', default=True)
    parser.add_argument('-r', dest='rnn_model', default=False)
    parser.add_argument('-g', dest='multi_gpu', default=False)
    args = parser.parse_args()

    training_config_path = args.training_config_path
    model_number = args.model_number
    sample_only = args.sample_only
    use_valid = args.use_valid
    rnn_model = args.rnn_model
    multi_gpu = args.multi_gpu

    run_training = RunTraining(training_config_path, model_number, sample_only, multi_gpu, use_valid, rnn_model)

    run_training.run()

    """
    Usage : 
    
    Run a training with the default parameters, config and model :
    $ python run_training.py
    
    For details about parameters, config and models, please refer to the README.md
    """




