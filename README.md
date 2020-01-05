#### DEEP LEARNING PIPELINE TO TRAIN A NEURAL NETWORK


To install the packages:
```
$ pip install -r requirements.txt
```

##### 0. To Quickly run a training with the default model, configuration and arguments :
```
$ python3 run_training.py
```

##### Usage :

##### 1. Model Selection
In the file ```model.py```, choose the model you would like to run
by uncommenting the respective line. The models implementation
are in the folder ```models/```. A new model can be implemented 
inside the file ```./models/my_model.py```.


##### 2. Training Configuration

Configure the training parameters in the file ```training_config.json```

Tune the values according to your preferences.

__Please, do not change the keys__

The most important parameters to set are :

```json
{
  "path" : {
    "root_folder" : "../dataset/",
    "train_path" : "train/train/",
    "target_path" : "train_kaggle.csv",
    "test_path" : "test/test",
    "checkpoint_filepath" : "checkpoint/"
  },
   "fit" : {
    "epochs": 1,
    "batch_size" : 128,
    "verbose" : 2
  }
}
```

The other parameters relate to :
- The test and valid size of the dataset
- The option to load a previous model to continue training
- Compiling option for the model
- Callbacks configuration for :
    - Early Stopping 
    - Model Checkpoint
    - Reduce Learning Rate of plateau
    - Note that you can choose to deactivate any callback by setting 
    ```"activate" : False``` 
 
The goal of this config file is to allow the user to use the script and run training in many
different conditions, without having to touch the code.


##### 3. Run Training :

Once the model and the configuration are set, a training can be run with
the following command :

- Example using the default arguments:
```
$ python3 run_training.py
```

- Example setting all the arguments :
```
$ python3 run_training.py \
-p training_config.json \
-n 1 \
-s False \
-t True \
-r False 
```

- Arguments available :
   - ```-p training_config.json``` (string) --> training_config_path : path of the json containing the training
   config
   - ``` -n 1``` (int) --> model_number : number of the model according to the Training Log
   (to save the files)
   - ``` -s False``` (boolean) --> sample_only : load only 1000 sample to test if the model is correct,
   compiles and runs, to save time.
   - ``` -t True ``` (boolean) --> use_valid : use a part of the dataset as a validation dataset
   - ``` -r False``` (boolean) --> rnn_model : True if the model to run is __ONLY__ RNN without CNN, False otherwise. 
   The data to be fed into a RNN doesn't have the same shape as the data to be fed into a CNN.


##### 4. Retrieving the Results of the training :

At the end of the training, the checkpoint folder will contain 5 files :
        The filenames depends on the config 'model_checkpoint'-->'file_basename'
   - ```file_basename_{number}.h5``` : the keras checkpoint of the "best" model
   - ```file_basename_{number}.json ``` : a json file containing the description of the model
   - ```file_basename_{number}.csv``` : the history of the training (epochs, loss, val_loss...)
   - ```file_basename_{number}.txt``` : a file text containing the test accuracy
   - ```file_basename_{number}_prediction.csv``` : the results of the prediction of the best model
   - ```file_basename_{number}_summary.txt``` : Summary of the model compiled with the number
   of arguments and the input and output shape of each layer

##### 5. Ensemble Modelling

After retrieving many predictions from good models, it is possible to ensemble the models
by averaging the prediction. To do that, use the script ```ensemble.py```

At the bottom of the script, in the ``` main```, specify :
- ```folder```: the folder containing the prediction csv
- ```filenames```: a list containing the filename of the predictions to ensemble
- ```output_folder```: the folder to output the ensemble prediction csv
- ```output_filename```: the filename of the ensemble csv

##### OPTION 1 : To save the output of the console, run the training with :

```
$ python3 run_training.py -p path -n number -s False -g False -t True | tee -a path/checkpoint/file_basename_output.txt
```

##### OPTION 2 : To Specify which GPU to use on the machine :

identify its number and input it on line 7 of training_manager.py :

```
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # or "1", or "0,1" if multiple GPUs are available
```
