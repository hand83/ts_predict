import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from abc import ABCMeta, abstractmethod

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
#config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


def signed_mse(penalty=0.0):
    """Create a modified mean-squared error loss function penalting predictions with opposite sign compared to target.
    
    :param float penalty: Penalty multiplier by wich the MSE will be increased as (1 + penalty) * MSE
    :return function:
    """

    def loss(y_actual, y_pred):
        is_opposite = 1 - (K.abs(K.sign(y_actual) + K.sign(y_pred)) / 2)
        mse = K.mean(K.square(y_pred - y_actual))
        return (1 + is_opposite * penalty) * mse

    return loss


class BaseNNModel(object, metaclass=ABCMeta):
    """Abstract class for time-series predicting sequential keras models"""
    
    def __init__(self, n_input, n_output, output_layer_properties=None, batch_size=200, epochs=100, 
                 optimizer_params=None, seed=None):
        self._n_input = n_input
        self._n_output = n_output
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        self.optimizer_params = optimizer_params or {}
        if self.seed:
            self.initializer = tf.keras.initializers.glorot_uniform(seed=self.seed)
        else:
            self.initializer = tf.keras.initializers.glorot_uniform()
        self.output_layer_properties = output_layer_properties or {}
        self._model = None
        self._history = None

    @abstractmethod
    def _construct_layers(self):
        pass

    def build_model(self, loss='mse'):
        """Clear keras session and compile model
        
        :param str,function loss: loss function to use, default: mse (mean squared error)
        """
        tf.keras.backend.clear_session()
        self._construct_layers()
        self._model.compile(optimizer=tf.keras.optimizers.Adam(**self.optimizer_params), loss=loss)
            
    def summary(self):
        return self._model.summary()
        
        
    def fit(self, x, y, validation_data=None):
        if self.seed:
            np.random.seed(self.seed)
            tf.random.set_seed(self.seed)
        self._history = self._model.fit(x=x, y=y, validation_data=validation_data,
                                       batch_size=self.batch_size, epochs=self.epochs,
                                       verbose=0)
                                       
    def plot_history(self, save_to=None):
        """Plot training history
        
        :param str save_to: filename to save, default: None - no saving, the AxesSubplot object is returned
        """
        if save_to:
            ax = pd.DataFrame.from_dict(self._history.history).plot()
            fig = ax.get_figure()
            fig.savefig(fname=save_to)
        else:
            return pd.DataFrame.from_dict(self._history.history).plot()
            
    def predict(self, x):
        return self._model.predict(x)
        
    def mse(self, x, y):
        """Mean squared error calculation
        
        :param numpy.Array x: input data for prediction
        :param numpy.Array y: target data
        :return float: mean squared error
        """
        return np.mean(np.square(self.predict(x) - y))
        
        
class ParamExtractMixin(object, metaclass=ABCMeta):
    """Mixin class with parameter extraction functions for Dense, 1D Convolutional and LSTM layers"""
    
    def _extract_dense_params(self, param_dict):
        # dropout parameters
        dropout = param_dict.get('dropout')
        if dropout:
            dropout_params = {'rate': dropout}
            if self.seed:
                dropout_params.update({'seed': self.seed})
        else:
            dropout_params = {}
        # dense layer parameters
        l1 = param_dict.get('l1') or 0.0
        l2 = param_dict.get('l2') or 0.0
        units = param_dict.get('units')
        activation = param_dict.get('activation') or 'relu'
        layer_params = {'units': int(units),
                        'activation': activation,
                        'kernel_initializer': self.initializer,
                        'kernel_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2)}
        return (dropout_params, layer_params)

    def _extract_conv_params(self, param_dict):
        # dropout parameters
        dropout = param_dict.get('dropout')
        if dropout:
            dropout_params = {'rate': dropout}
            if self.seed:
                dropout_params.update({'seed': self.seed})
        else:
            dropout_params = {}
        # Conv1D layer parameters
        l1 = param_dict.get('l1') or 0.0
        l2 = param_dict.get('l2') or 0.0
        filters = param_dict.get('filters')
        kernel_size = param_dict.get('kernel_size')
        padding = param_dict.get('padding') or 'valid'
        activation = param_dict.get('activation') or 'relu'
        layer_params = {'filters': int(filters),
                        'kernel_size': int(kernel_size),
                        'activation': activation,
                        'padding': padding,
                        'kernel_initializer': self.initializer,
                        'kernel_regularizer': tf.keras.regularizers.L1L2(l1=l1, l2=l2)}
        # AveragePooling1D layer parameters
        pool_size = param_dict.get('pool_size')
        pool_stride = param_dict.get('pool_stride')
        if not pd.isna(pool_size):
            pool_params = {'pool_size': int(pool_size), 
                           'strides': None if pd.isna(pool_stride) else int(pool_stride)}
        else:
            pool_params = {}
        return (dropout_params, layer_params, pool_params)
        
    def _extract_lstm_params(self, param_dict):
        # LSTM layer parameters
        units = param_dict.get('units')
        kernel_l1 = param_dict.get('kernel_l1') or 0.0
        kernel_l2 = param_dict.get('kernel_l2') or 0.0
        recurrent_l1 = param_dict.get('recurrent_l1') or 0.0
        recurrent_l2 = param_dict.get('recurrent_l2') or 0.0
        kernel_dropout = param_dict.get('kernel_dropout') or 0.0
        recurrent_dropout = param_dict.get('recurrent_dropout') or 0.0
        return_sequences = param_dict.get('return_sequences') or True
        activation = param_dict.get('activation') or 'tanh'
        recurrent_activation = param_dict.get('recurrent_activation') or 'sigmoid'
        layer_params = {'units': int(units),
                        'return_sequences': return_sequences,
                        'activation': activation,
                        'recurrent_activation': recurrent_activation,
                        'dropout': kernel_dropout,
                        'recurrent_dropout': recurrent_dropout,
                        'kernel_initializer': self.initializer,
                        'kernel_regularizer': tf.keras.regularizers.L1L2(l1=kernel_l1, l2=kernel_l2),
                        'recurrent_regularizer': tf.keras.regularizers.L1L2(l1=recurrent_l1, l2=recurrent_l2)}
        return layer_params
    

class FFNNModel(BaseNNModel, ParamExtractMixin):
    """Class for time-series predicting Fill-forward sequential keras model"""

    def __init__(self, n_input, n_output, hidden_layer_properties, output_layer_properties=None,
                 batch_size=200, epochs=100, optimizer_params=None, seed=None):
        """FFNNModel initializer
        
        :param int n_input: time series history length to use for prediction
        :param int n_output: number of look-ahead predictions
        :param dict,list hidden_layer_properties: dictionary or list of dictionaries with hidden fillforward layer 
        parameters such as units, l1, l2, activation, dropout
        :param dict output_layer_properties: dictionary output layer parameters such as l1, l2, dropout
        :param int batch_size: batch size for fitting, default: 200
        :param int epochs: number of epochs for fitting, default: 100
        :param dict optimizer_params: custom optimizer arguments passed to Adam optimizer
        :param int seed: random seed for reproducibility, default: None - no seed used
        """
        super(FFNNModel, self).__init__(n_input, n_output, output_layer_properties, batch_size, epochs, 
                                        optimizer_params, seed)
        if isinstance(hidden_layer_properties, list):
            self.hidden_layer_properties = hidden_layer_properties
        else:
            self.hidden_layer_properties = [hidden_layer_properties]  # single dict as one hidden layer
        
    def _construct_layers(self):
        self._model = tf.keras.Sequential()
        # input layer
        self._model.add(tf.keras.layers.Input(shape=(self._n_input,)))
        # hidden layers
        for param in self.hidden_layer_properties:
            dp, lp = self._extract_dense_params(param)
            if dp:
                self._model.add(tf.keras.layers.Dropout(**dp))
            self._model.add(tf.keras.layers.Dense(**lp))
        # output layers
        output_layer_properties = self.output_layer_properties.copy()
        if 'activation' not in output_layer_properties:
            # set default output activation function to linear
            output_layer_properties['activation'] = 'linear'
        output_layer_properties['units'] = self._n_output
        dp, lp = self._extract_dense_params(output_layer_properties)
        if dp:
            self._model.add(tf.keras.layers.Dropout(**dp))
        self._model.add(tf.keras.layers.Dense(**lp))
        
        
class CNNModel(BaseNNModel, ParamExtractMixin):
    """Class for time-series predicting 1D convolutional sequential keras model"""
    
    def __init__(self, n_input, n_output, hidden_layer_properties, output_layer_properties=None,
                 batch_size=200, epochs=100, optimizer_params=None, seed=None):
        """CNNModel initializer
        
        :param int n_input: time series history length to use for prediction
        :param int n_output: number of look-ahead predictions
        :param dict,list hidden_layer_properties: dictionary or list of dictionaries with hidden 1D convolutional 
        layer parameters such as filter, kernel_size, padding, l1, l2, dropout, pool_size
        :param dict output_layer_properties: dictionary output layer parameters such as l1, l2, dropout
        :param int batch_size: batch size for fitting, default: 200
        :param int epochs: number of epochs for fitting, default: 100
        :param dict optimizer_params: custom optimizer arguments passed to Adam optimizer
        :param int seed: random seed for reproducibility, default: None - no seed used
        """
        super(CNNModel, self).__init__(n_input, n_output, output_layer_properties, batch_size, epochs, 
                                        optimizer_params, seed)
        if isinstance(hidden_layer_properties, list):
            self.hidden_layer_properties = hidden_layer_properties
        else:
            self.hidden_layer_properties = [hidden_layer_properties]  # single dict as one hidden layer
    
    def _construct_layers(self):
        self._model = tf.keras.Sequential()
        # input layer
        self._model.add(tf.keras.layers.Input(shape=(self._n_input,)))
        self._model.add(tf.keras.layers.Reshape((self._n_input, 1)))
        # hidden layers
        for param in self.hidden_layer_properties:
            dp, lp, pp = self._extract_conv_params(param)
            if dp:
                self._model.add(tf.keras.layers.Dropout(**dp))
            self._model.add(tf.keras.layers.Conv1D(**lp))
            if pp:
                self._model.add(tf.keras.layers.AveragePooling1D(**pp))
        # output layers
        self._model.add(tf.keras.layers.Flatten())
        output_layer_properties = self.output_layer_properties.copy()
        if 'activation' not in output_layer_properties:
            # set default output activation function to linear
            output_layer_properties['activation'] = 'linear'
        output_layer_properties['units'] = self._n_output
        dp, lp = self._extract_dense_params(output_layer_properties)
        if dp:
            self._model.add(tf.keras.layers.Dropout(**dp))
        self._model.add(tf.keras.layers.Dense(**lp))    
        
        
class LSTMModel(BaseNNModel, ParamExtractMixin):
    """Class for time-series predicting LSTM sequential keras model"""
    
    def __init__(self, n_input, n_output, hidden_layer_properties, output_layer_properties=None,
                 batch_size=200, epochs=100, optimizer_params=None, seed=None):
        """LSTMModel initializer
        
        :param int n_input: time series history length to use for prediction
        :param int n_output: number of look-ahead predictions
        :param dict,list hidden_layer_properties: dictionary or list of dictionaries with hidden LSTM 
        layer parameters such as units, kernel_dropout, recurrent_dropout, kernel_l1, kernel_l2, recurrent_l1, 
        recurrent_l2, return_sequences
        :param dict output_layer_properties: dictionary output layer parameters such as l1, l2, dropout
        :param int batch_size: batch size for fitting, default: 200
        :param int epochs: number of epochs for fitting, default: 100
        :param dict optimizer_params: custom optimizer arguments passed to Adam optimizer
        :param int seed: random seed for reproducibility, default: None - no seed used
        """
        super(LSTMModel, self).__init__(n_input, n_output, output_layer_properties, batch_size, epochs, 
                                        optimizer_params, seed)
        if isinstance(hidden_layer_properties, list):
            self.hidden_layer_properties = hidden_layer_properties
        else:
            self.hidden_layer_properties = [hidden_layer_properties]  # single dict as one hidden layer
    
    def _construct_layers(self):
        self._model = tf.keras.Sequential()
        # input layer
        self._model.add(tf.keras.layers.Input(shape=(self._n_input,)))
        self._model.add(tf.keras.layers.Reshape((self._n_input, 1)))
        # hidden layers
        for i, param in enumerate(self.hidden_layer_properties):
            lstm_param = param.copy()
            if i < len(self.hidden_layer_properties) - 1:
                # make sure that chained LSTM layers pass sequences to each other
                lstm_param['return_sequences'] = True
            lp = self._extract_lstm_params(lstm_param)
            self._model.add(tf.keras.layers.LSTM(**lp))
        # output layers
        if lp['return_sequences']:
            self._model.add(tf.keras.layers.Flatten())
        output_layer_properties = self.output_layer_properties.copy()
        if 'activation' not in output_layer_properties:
            # set default output activation function to linear
            output_layer_properties['activation'] = 'linear'
        output_layer_properties['units'] = self._n_output
        dp, lp = self._extract_dense_params(output_layer_properties)
        if dp:
            self._model.add(tf.keras.layers.Dropout(**dp))
        self._model.add(tf.keras.layers.Dense(**lp))    
    
    
class ConvLSTMModel(BaseNNModel, ParamExtractMixin):
    """Class for time-series predicting combined Convolutional-LSTM sequential keras model"""
    
    def __init__(self, n_input, n_output, conv_layer_properties, lstm_layer_properties, output_layer_properties=None,
                 batch_size=200, epochs=100, optimizer_params=None, seed=None):
        """ConvLSTMModel initializer
        
        :param int n_input: time series history length to use for prediction
        :param int n_output: number of look-ahead predictions
        :param dict conv_layer_properties: dictionary with hidden 1D convolutional 
        layer parameters such as filter, kernel_size, padding, l1, l2, dropout, pool_size
        :param dict lstm_layer_properties: dictionary with hidden LSTM 
        layer parameters such as units, kernel_dropout, recurrent_dropout, kernel_l1, kernel_l2, recurrent_l1, 
        recurrent_l2, return_sequences
        :param dict output_layer_properties: dictionary output layer parameters such as l1, l2, dropout
        :param int batch_size: batch size for fitting, default: 200
        :param int epochs: number of epochs for fitting, default: 100
        :param dict optimizer_params: custom optimizer arguments passed to Adam optimizer
        :param int seed: random seed for reproducibility, default: None - no seed used
        """
        super(ConvLSTMModel, self).__init__(n_input, n_output, output_layer_properties, batch_size, epochs, 
                                            optimizer_params, seed)
        self.conv_layer_properties = conv_layer_properties
        self.lstm_layer_properties = lstm_layer_properties
    
    def _construct_layers(self):
        self._model = tf.keras.Sequential()
        # input layer
        self._model.add(tf.keras.layers.Input(shape=(self._n_input,)))
        self._model.add(tf.keras.layers.Reshape((self._n_input, 1)))
        # hidden layers - convolutional layer
        dp, lp, pp = self._extract_conv_params(self.conv_layer_properties)
        if dp:
            self._model.add(tf.keras.layers.Dropout(**dp))
        self._model.add(tf.keras.layers.Conv1D(**lp))
        if pp:
            self._model.add(tf.keras.layers.AveragePooling1D(**pp))
        # hidden layers - LSTM layer
        lp = self._extract_lstm_params(self.lstm_layer_properties)
        self._model.add(tf.keras.layers.LSTM(**lp))
        # output layers
        if lp['return_sequences']:
            self._model.add(tf.keras.layers.Flatten())
        output_layer_properties = self.output_layer_properties.copy()
        if 'activation' not in output_layer_properties:
            # set default output activation function to linear
            output_layer_properties['activation'] = 'linear'
        output_layer_properties['units'] = self._n_output
        dp, lp = self._extract_dense_params(output_layer_properties)
        if dp:
            self._model.add(tf.keras.layers.Dropout(**dp))
        self._model.add(tf.keras.layers.Dense(**lp))        
