import os
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import dask
from dask.distributed import Client


class ESM(object):
    
    def __init__(self, trend, damped, seasonal=None, periods=None, use_boxcox=False, remove_bias=False):
            self.trend = trend
            self.damped = damped
            self.seasonal = seasonal
            self.periods = periods
            self.use_boxcox = use_boxcox
            self.remove_bias = remove_bias
            
    def predict(self, history, steps=1):
        try:
            model = ExponentialSmoothing(history, trend=self.trend, damped=self.damped, seasonal=self.seasonal, 
                                             seasonal_periods=self.periods)
            model_fit = model.fit(optimized=True, use_boxcox=self.use_boxcox, remove_bias=self.remove_bias)
            yhat = model_fit.forecast(steps)
        except:
            yhat = np.full((steps,), np.NaN)
        return yhat
    
    def bulk_predict(self, X, steps):
        yhat_list = []
        for x in X:
            yhat = self.predict(x, steps)
            yhat_list.append(yhat)
        return np.array(yhat_list)
    
    def stepforward_predict(self, x, initial_history, fixed_history=False):
        pred_y = []
        if fixed_history:
            for i in range(x.shape[0] - initial_history):
                yhat = self.predict(x[i:i+initial_history])
                pred_y.append(yhat[0])
        else:
            for i in range(initial_history, x.shape[0]):
                yhat = self.predict(x[:i], steps=1)
                pred_y.append(yhat[0])
        return np.array(pred_y)
        
        
def validate_esm_params(params, ts, initial_history, fixed_history=False):
    esm = ESM(**params)
    pred_y = esm.stepforward_predict(ts, initial_history, fixed_history)
    n_failures = (np.isnan(pred_y)).sum()
    mse = np.nanmean(np.square(pred_Y - ts[initial_history:]))
    return (mse, n_failures)
    
    
def esm_grid_search(ts, initial_history, fixed_history, params_list, parallel=False):
    if parallel:
        client = Client()
        #tasks = [client.submit(func=validate_esm_params, 
        #                       params=params, ts=ts, inital_history=inital_history, fixed_history=fixed_history) 
        #         for params in params_list]
        #results = [t.result() for t in tasks]
        tasks = []
        for params in parms_list:
            block = dask.delayed(validate_esm_params)(params, ts, initial_history, fixed_history)
            tasks.append(block)
        results = dask.compute(*tasks)
        client.close()
    else:
        results = [validate_esm_params(params, ts, initial_history, fixed_history)]
    mse_list, failure_list = zip(*results)
    mse_list = np.concatenate(mse_list, axis=0)
    failure_list = np.concatenate(failure_list, axis=0)
    mse_df = pd.DataFrame(mse_list).rename(columns={x: 'mse_{}'.format(x) for x in range(mse_list.shape[0])})
    failure_df = pd.DataFrame(failure_list).rename(columns={x: 'failures_{}'.format(x) for x in 
                                                            range(failure_list.shape[0])})
    params_df = pd.DataFrame(params_list)
    return pd.concat([params_df, mse_df, failure_df], axis=1)
    
    
def esm_validation_runner(hyperparam_path, data_path, valid_date_range, history_limit=None, fixed_history=False, 
                          diff=False, result_dir='./', parallel=False):
    # load hyperparameters
    params_list = pd.read_csv(hyperparam_path).to_dict(orient='records')
    # load data
    ts = pd.read_csv(data_path, index_col=0, parse_dates=True)
    ts = ts.loc[~ts.index.duplicated(), 'Close']
    ts = ts.loc[ts.index < valid_date_range[1]].dropna()
    if diff:
        ts = ts.diff().dropna()
    initial_history = ts.loc[ts.index < valid_date_range[0]].shape[0]
    if history_limit is not None:
        ts = ts.iloc[initial_history - history_limit:]
        initial_history = history_limit
    result = esm_grid_search(ts.values, initial_history, fixed_history, params_list, parallel)
    result_path = os.path.join(result_dir, 'esm{:d}_{}_{}_{}.csv').format(
        initial_history, fixed_history,
        os.path.basename(hyperparam_path).split('.')[0],
        pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')
    )
    result.to_csv(result_path, index=False)  
    
    