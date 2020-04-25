from functools import wraps
from itertools import product
import pandas as pd
import numpy as np
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.preprocessing import StandardScaler


def generate_param_list(param_items):
    """Generate parameters list from the product of parameter items.
    
    :param dict param_items: Dictionary with parameter name as key and candidate parameter list as values
    :return list: list of dictionaries with the cartesian product of parameter items
    """
    prod = product(*(param_items[p] for p in param_items))
    return [dict(zip(param_items.keys(), x)) for x in prod]
    

def get_partitions(ts, split_pct):
    split_ndx = int(np.ceil(ts.shape[0] * split_pct))
    if isinstance(ts, pd.core.series.Series):
        partitions = ts.iloc[:split_ndx], ts.iloc[split_ndx:]
    else:
        partitions = ts[:split_ndx], ts[split_ndx:]
    return partitions


def get_rolling_windows(ts, window, return_indices=False):
    if isinstance(ts, pd.core.series.Series):
        M = np.array([ts.shift(x).values for x in range(window)]).T
        ndx = ts.index[window - 1:]
    else:
        M = np.array([pd.Series(ts).shift(x).values for x in range(window)]).T
        ndx = np.arange(window - 1, ts.shape[0])
    M = M[window-1:, ::-1]
    if return_indices:
        res = M, ndx
    else:
        res = M
    return res
    

def ts_process(shape=(-1,)):
    def ts_process_actual(fun):
        @wraps(fun)
        def wrapper(obj, ts, *args, **kwargs):
            if isinstance(ts, pd.core.series.Series):
                res = fun(obj, ts.values.reshape(shape), *args, **kwargs)
                if res is not None:
                    res = pd.Series(res.flatten(), index=ts.index)
            else:
                res = fun(obj, ts.reshape(shape), *args, **kwargs)
                if res is not None:
                    res = res.reshape(ts.shape)
            return res
        return wrapper
    return ts_process_actual


class StandardScaler1D(object):

    def __init__(self):
        self._scaler = StandardScaler()
        
    @property
    def params(self):
        return {'mean': self._scaler.mean_[0], 'std': np.sqrt(self._scaler.var_[0])}
        
    @ts_process(shape=(-1, 1))
    def fit(self, ts):
        self._scaler.fit(ts.reshape(-1, 1))
    
    @ts_process(shape=(-1, 1))
    def transform(self, ts):
        return self._scaler.transform(ts)
    
    @ts_process(shape=(-1, 1))
    def fit_transform(self, ts):
        return self._scaler.fit_transform(ts)
    
    @ts_process(shape=(-1, 1))
    def inverse_transform(self, ts):
        return self._scaler.inverse_transform(ts)
        
        
class BoxCoxScaler(object):

    def __init__(self, offset=0.1):
        self._shift = None
        self._lambda = None
        self._offset = offset
       
    @property
    def params(self):
        return {'shift': self._shift, 'lambda': self._lambda}
        
    @ts_process()
    def fit(self, ts, min_val=None):
        _ = self._fit_transform(ts, min_val)

    @ts_process()
    def transform(self, ts):
        return boxcox(ts + self._shift, self._lambda)

    @ts_process()      
    def fit_transform(self, ts, min_val=None):
        return self._fit_transform(ts, min_val)
    
    def _fit_transform(self, ts, min_val=None):
        self._shift = self._offset + np.abs(min(0, min_val if min_val is not None else ts.min()))
        bc_ts, self._lambda = boxcox(ts + self._shift)
        return bc_ts
        
    @ts_process()
    def inverse_transform(self, ts):
        return inv_boxcox(ts, self._lambda) - self._shift
        
