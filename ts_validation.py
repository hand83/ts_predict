import os
import json
from itertools import product
import pandas as pd
import numpy as np
import dask
from dask.distributed import Client
from ts_ml import signed_mse, FFNNModel, CNNModel, LSTMModel, ConvLSTMModel
from ts_lib import get_rolling_windows, StandardScaler1D, BoxCoxScaler 


def validate_hyperparams(params, train_X, train_Y, valid_X, valid_Y, model_factory, scaler=None):
    model = model_factory.build_model(params)
    model.fit(x=train_X, y=train_Y)
    pred_Y = model.predict(valid_X)
    if scaler is not None:
        pred_Y = scaler.inverse_transform(pred_Y)
    mse = np.mean(np.square(pred_Y - valid_Y), axis=0)
    return mse


def grid_search_hyperparams(train_ts, valid_ts, n_input, n_output, model_factory, params_list, preprocess=False, 
                            parallel=False):
    # prepare data
    if preprocess:
        scaler = StandardScaler1D()
        train_ts = scaler.fit_transform(train_ts)
    else:
        scaler = None
    train_X = get_rolling_windows(train_ts, n_input + n_output)
    train_X, train_Y = train_X[:, :-n_output], train_X[:, -n_output:].reshape((-1, n_output))
    valid_X = get_rolling_windows(valid_ts, n_input + n_output)
    valid_X, valid_Y = valid_X[:, :-n_output], valid_X[:, -n_output:].reshape((-1, n_output))
    if preprocess:
        valid_X = scaler.transform(valid_X)
    # run models
    if parallel:
        # Keras models are not thread-safe
        raise NotImplementedError
    else:
        results = [validate_hyperparams(params, train_X, train_Y, valid_X, valid_Y, model_factory, scaler) 
                   for params in params_list]
    # collect results
    mse_list = np.concatenate(results, axis=0)
    mse_df = pd.DataFrame(mse_list).rename(columns={x: 'mse_{}'.format(x) for x in range(mse_list.shape[0])})
    param_df = pd.DataFrame(params_list)
    return pd.concat([param_df, mse_df], axis=1)

    
def nn_validation_runner(model_config_path, data_path, valid_date_range, seed, hyperparam_path, result_dir='./'):
    # load hyperparameters
    params_list = pd.read_csv(hyperparam_path).to_dict(orient='records')
    # load data
    ts = pd.read_csv(data_path, index_col=0, parse_dates=True)
    ts = ts.loc[~ts.index.duplicated(), 'Close']
    train_ts = ts.loc[ts.index < valid_date_range[0]].diff().dropna()
    valid_ts = ts.loc[(ts.index >= valid_date_range[0]) & (ts.index < valid_date_range[1])].diff().dropna()
    # load model config
    factory_mapping = {x.__name__: x for x in [FFNModelFactory, CNNModelFactory, LSTMModelFactory, 
                                               ConvLSTMModelFactory]}
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)
    n_input = model_config['factory_params']['fixed_model_params']['n_input']
    n_output = model_config['factory_params']['fixed_model_params']['n_output']
    preprocess = model_config.get('preprocess') or False
    model_factory = factory_mapping[model_config['model_factory']](seed=seed, **model_config['factory_params'])
    result = grid_search_hyperparams(train_ts, valid_ts, n_input, n_output, model_factory, params_list, preprocess)
    result_path = os.path.join(result_dir, '{}_{}_{}.csv').format(os.path.basename(model_config_path).split('.')[0],
                                                                  os.path.basename(hyperparam_path).split('.')[0],
                                                                  pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S'))
    result.to_csv(result_path, index=False)


class ModelFactory(object):

    model_param_list = ['n_input', 'n_output', 'batch_size', 'epochs']
    build_param_list = ['loss']
    adam_param_list = ['learning_rate', 'beta_1', 'beta_2', 'clipnorm', 'clipvalue']
    dense_mapping = {'hidden_units': 'units',
                     'hidden_activation': 'activation',
                     'hidden_l1': 'l1',
                     'hidden_l2': 'l2',
                     'hidden_dropout': 'dropout'}
    conv_mapping = {'filters': 'filters',
                    'kernel_size': 'kernel_size',
                    'padding': 'padding',
                    'pool_size': 'pool_size',
                    'pool_stride': 'pool_stride',
                    'conv_activation': 'activation',
                    'conv_l1': 'l1',
                    'conv_l2': 'l2',
                    'conv_dropout': 'dropout'}
    lstm_mapping = {'recurrent_activation': 'recurrent_activation',
                    'kernel_l1': 'kernel_l1',
                    'kernel_l2': 'kernel_l2',
                    'recurrent_l1': 'recurrent_l1',
                    'recurrent_l2': 'recurrent_l2',
                    'kernel_dropout': 'kernel_dropout',
                    'recurrent_dropout': 'recurrent_dropout',
                    'return_sequences': 'return_sequences',
                    'lstm_units': 'units',
                    'lstm_activation': 'activation'}
    output_layer_mapping = {'output_activation': 'activation',
                            'output_l1': 'l1',
                            'output_l2': 'l2',
                            'output_dropout': 'dropout'}
    
    def __init__(self, seed, fixed_model_params, fixed_output_layer_params=None, fixed_adam_parameters=None, 
                fixed_build_params=None,):
        self.fixed_model_params = fixed_model_params
        self.fixed_model_params.update({'seed': seed})
        self.fixed_output_layer_params = fixed_output_layer_params or {}
        self.fixed_adam_parameters = fixed_adam_parameters or {}
        self.fixed_build_params = fixed_build_params or {}

 
class FFNModelFactory(ModelFactory):
    
    def __init__(self, seed, fixed_model_params, fixed_hidden_layer_params=None, fixed_output_layer_params=None, 
                 fixed_build_params=None):
        super(FFNModelFactory, self).__init__(seed, fixed_model_params, fixed_output_layer_params, 
                                              fixed_build_params)
        self.fixed_hidden_layer_params = fixed_hidden_layer_params or {}
                 
    def map_params(self, params):
        model_params = {k: v for k, v in params.items() if k in self.model_param_list}
        build_params = {k: signed_mse(v) for k, v in params.items() if k in self.build_param_list}
        adam_params = {k: v for k, v in params.items() if k in self.adam_param_list}
        hidden_layer_params = {self.dense_mapping[k]: v for k, v in params.items() if k in 
                               self.dense_mapping}
        output_layer_params = {self.output_layer_mapping[k]: v for k, v in params.items() if k in 
                               self.output_layer_mapping}
        model_params.update(self.fixed_model_params)
        build_params.update(self.fixed_build_params)
        hidden_layer_params.update(self.fixed_hidden_layer_params)
        output_layer_params.update(self.fixed_output_layer_params)
        return (build_params, model_params, hidden_layer_params, output_layer_params, adam_params)
                
    def build_model(self, params):
        build_params, model_params, hidden_layer_params, output_layer_params, adam_params = self.map_params(params)
        model = FFNNModel(**model_params, hidden_layer_properties=hidden_layer_params, 
                          output_layer_properties=output_layer_params, optimizer_params=adam_params)
        model.build_model(**build_params)
        return model
        
        
class CNNModelFactory(ModelFactory):
    
    def __init__(self, seed, fixed_model_params, fixed_hidden_layer_params=None, fixed_output_layer_params=None, 
                 fixed_build_params=None):
        super(CNNModelFactory, self).__init__(seed, fixed_model_params, fixed_output_layer_params, 
                                              fixed_build_params)
        self.fixed_hidden_layer_params = fixed_hidden_layer_params or {}
                 
    def map_params(self, params):
        model_params = {k: v for k, v in params.items() if k in self.model_param_list}
        build_params = {k: signed_mse(v) for k, v in params.items() if k in self.build_param_list}
        adam_params = {k: v for k, v in params.items() if k in self.adam_param_list}
        hidden_layer_params = {self.conv_mapping[k]: v for k, v in params.items() if k in 
                               self.conv_mapping}
        output_layer_params = {self.output_layer_mapping[k]: v for k, v in params.items() if k in 
                               self.output_layer_mapping}
        model_params.update(self.fixed_model_params)
        build_params.update(self.fixed_build_params)
        hidden_layer_params.update(self.fixed_hidden_layer_params)
        output_layer_params.update(self.fixed_output_layer_params)
        return (build_params, model_params, hidden_layer_params, output_layer_params, adam_params)
                
    def build_model(self, params):
        build_params, model_params, hidden_layer_params, output_layer_params, adam_params = self.map_params(params)
        model = CNNModel(**model_params, hidden_layer_properties=hidden_layer_params, 
                          output_layer_properties=output_layer_params, optimizer_params=adam_params)
        model.build_model(**build_params)
        return model
        
        
class LSTMModelFactory(ModelFactory):
    
    def __init__(self, seed, fixed_model_params, fixed_hidden_layer_params=None, fixed_output_layer_params=None, 
                 fixed_build_params=None):
        super(LSTMModelFactory, self).__init__(seed, fixed_model_params, fixed_output_layer_params, 
                                              fixed_build_params)
        self.fixed_hidden_layer_params = fixed_hidden_layer_params or {}
                 
    def map_params(self, params):
        model_params = {k: v for k, v in params.items() if k in self.model_param_list}
        build_params = {k: signed_mse(v) for k, v in params.items() if k in self.build_param_list}
        adam_params = {k: v for k, v in params.items() if k in self.adam_param_list}
        hidden_layer_params = {self.lstm_mapping[k]: v for k, v in params.items() if k in 
                               self.lstm_mapping}
        output_layer_params = {self.output_layer_mapping[k]: v for k, v in params.items() if k in 
                               self.output_layer_mapping}
        model_params.update(self.fixed_model_params)
        build_params.update(self.fixed_build_params)
        hidden_layer_params.update(self.fixed_hidden_layer_params)
        output_layer_params.update(self.fixed_output_layer_params)
        return (build_params, model_params, hidden_layer_params, output_layer_params, adam_params)
                
    def build_model(self, params):
        build_params, model_params, hidden_layer_params, output_layer_params, adam_params = self.map_params(params)
        model = LSTMModel(**model_params, hidden_layer_properties=hidden_layer_params, 
                          output_layer_properties=output_layer_params, optimizer_params=adam_params)
        model.build_model(**build_params)
        return model
        
        
class ConvLSTMModelFactory(ModelFactory):
    
    def __init__(self, seed, fixed_model_params, fixed_conv_params=None, fixed_lstm_params=None, 
                 fixed_output_layer_params=None, fixed_build_params=None):
        super(ConvLSTMModelFactory, self).__init__(seed, fixed_model_params, fixed_output_layer_params, 
                                                   fixed_build_params)
        self.fixed_conv_params = fixed_conv_params or {}
        self.fixed_lstm_params = fixed_lstm_params or {}
                 
    def map_params(self, params):
        model_params = {k: v for k, v in params.items() if k in self.model_param_list}
        build_params = {k: signed_mse(v) for k, v in params.items() if k in self.build_param_list}
        adam_params = {k: v for k, v in params.items() if k in self.adam_param_list}
        conv_params = {self.conv_mapping[k]: v for k, v in params.items() if k in self.conv_mapping}
        lstm_params = {self.lstm_mapping[k]: v for k, v in params.items() if k in self.lstm_mapping}
        output_layer_params = {self.output_layer_mapping[k]: v for k, v in params.items() if k in 
                               self.output_layer_mapping}
        model_params.update(self.fixed_model_params)
        build_params.update(self.fixed_build_params)
        conv_params.update(self.fixed_conv_params)
        lstm_params.update(self.fixed_lstm_params)
        output_layer_params.update(self.fixed_output_layer_params)
        return (build_params, model_params, conv_params, lstm_params, output_layer_params, adam_params)
                
    def build_model(self, params):
        build_params, model_params, conv_params, lstm_params, output_layer_params, adam_params = self.map_params(params)
        model = ConvLSTMModel(**model_params, output_layer_properties=output_layer_params,
                              conv_layer_properties=conv_params, lstm_layer_properties=lstm_params, 
                              optimizer_params=adam_params)
        model.build_model(**build_params)
        return model
        