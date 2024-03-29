import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin

class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, sample_param_here="Not using this"):
        self.sample_param_here = sample_param_here
            
    def fit(self, X, y):
        self._mean = y.mean()
        
        return self
    
    def predict(self, X):
        return np.ones((len(X),))*self._mean
