import os
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import dask
from dask.distributed import Client
from ts_lib import get_rolling_windows, BoxCoxScaler


class SARIMAModel(object):

    def __init__(self, p=1, d=0, q=0, seasonal_p=0, seasonal_d=0, seasonal_q=0, periodicity=0, trend=None):
        self.order = (p, d, q)
        self.seasonal_order = (seasonal_p, seasonal_d, seasonal_q, periodicity)
        self.trend = trend
        
    def aic_from_fit(self, x):
        try:
            model = SARIMAX(x, order=self.order, seasonal_order=self.seasonal_order, trend=self.trend, 
                           simple_differencing=True)
            model_fit = model.fit(disp=False)
            aic = model_fit.aic
        except:
            aic = None
        return aic

    def predict(self, x, steps=1):
        try:
            model = SARIMAX(x, order=self.order, seasonal_order=self.seasonal_order, trend=self.trend, 
                            simple_differencing=True)
            model_fit = model.fit(disp=False)
            yhat = model_fit.forecast(steps)
        except:
            yhat = np.full((steps,), np.NaN)
        return yhat
        
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
        pred_y = np.array(pred_y)
        return pred_y
        
        
def grid_search_sarima_aic(param_list, data):
    aic_list = []
    for param in param_list:
        model = SARIMAModel(**param)
        aic_list.append(model.aic_from_fit(data))
    param_df = pd.DataFrame(param_list)
    param_df = pd.concat([param_df, pd.Series(aic_list, name='AIC')], axis=1).sort_values('AIC')
    return param_df
