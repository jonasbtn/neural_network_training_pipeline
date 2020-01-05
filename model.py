from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, BatchNormalization, Bidirectional, Activation, Permute, Reshape
from keras.layers.recurrent import LSTM

from models.model_24 import Model_24
from models.model_27 import Model_27
from models.model_28 import Model_28
from models.cnn_rnn_model import cnn_rnn_model
from models.rnn_baseline import rnn_baseline
from models.rnn_best import rnn_best
from models.rnn_batchnorm import rnn_batchnorm
from models.my_model import My_Model

def Model():

    # # Call a model already implemented in another file :
    # # Uncomment the line of the model you want to choose

    model = Model_24()
    # model = Model_27()
    # model = Model_28()
    # model = rnn_baseline()
    # model = rnn_batchnorm()
    # model = rnn_best()
    # model = cnn_rnn_model()

    # # Or implement a new model inside the file "models/my_model.py"
    # model = My_Model()

    return model
