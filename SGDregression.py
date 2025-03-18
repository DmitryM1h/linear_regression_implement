import numpy as np
import pandas as pd
from LinearRegression import Regression

class SGDRegressor(Regression):
    def fit(self,X:np.array,y:np.array):
        if X.shape[0] != y.shape[0]:
            raise ValueError(f'Inconsistent shapes of X and y: {X.shape[0]} and {y.shape[0]}')
        if X.size == 0 or y.size == 0:
            raise ValueError('Input arrays X and y cannot be empty')
        self.weights_ = np.zeros(X.shape[1]+1)
        X = np.insert(X,0,1,axis=1)
        while(True):
            self.iters_done += 1
            self.weights_history.append(self.weights_)
            self.mse_.append(self._countMSE(X,y,self.weights_))
            num = np.random.randint(X.shape[0])
            error = y[num]-np.dot(self.weights_,X[num])
            grad = X[num] * -2 * error
            if any(np.abs(grad)<10**-5):
                self.iter = iter
                break
            self.weights_ = self.weights_ - self.eta*grad