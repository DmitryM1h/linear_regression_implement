import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
class Regression(BaseEstimator):
    def __init__(self,*,eta=0.001,n_iter=10):
        self.mse_ = []
        self.weights_ = []
        self.weights_history = []
        self.n_iter = n_iter
        self.eta = eta
    
    def _grad(self,X:np.array,y:np.array,w:np.array):
        '''
        N = X.shape[0]
        M = X.shape[1]
        grad = np.zeros(M)
        for col in range(M):
            s=0
            for obs in range(N):
                s += (y[obs] - np.dot(w,X[obs])) * X[obs][col]
            s = -2 * s
            grad[col] = s
        return grad/N '''
        N = X.shape[0]  
        predictions = np.dot(X, w)
        errors = y - predictions
        grad = -2 / N * np.dot(X.T, errors)
        return grad
        
    def __countMSE(self,X:np.array,y:np.array,w:np.array):
        N = X.shape[0]
        MSE = 0
        for i in range(N):
            MSE += np.square(y[i] - np.dot(w,X[i]))
        return MSE/N
    
    
    def fit(self,X:np.array,y:np.array):
        if X.shape[0] != y.shape[0]:
            raise ValueError(f'Inconsistent shapes of X and y: {X.shape[0]} and {y.shape[0]}')
        if X.size == 0 or y.size == 0:
            raise ValueError('Input arrays X and y cannot be empty')
        self.weights_ = np.zeros(X.shape[1]+1)
        X = np.insert(X,0,1,axis=1)
        for _ in range(self.n_iter):
            self.weights_history.append(self.weights_)
            self.mse_.append(self.__countMSE(X,y,self.weights_))
            self.weights_ = self.weights_ - self.eta*self._grad(X,y,self.weights_)
            
    def predict(self,X:np.array):
        X = np.insert(X,0,1,axis=1)
        return np.dot(X,self.weights_)