import numpy as np

from sklearn.preprocessing import StandardScaler
from penlm.base_estimators import BaseRegressor


class RidgeRegressor(BaseRegressor):
    def __init__(self,
                 fit_intercept: bool = True,
                 scale: bool = True):
        self.fit_intercept = fit_intercept
        self.parameters = {}
        self.lambd = None
        self.beta = None
        self.intercept = None
        self.scale = scale
        self.scaler = None

    
    def set_parameters(self,
                       parameters: dict):
        self.lambd = parameters['lambda']
        self.parameters['lambda'] = self.lambd


    def fit(self,
            X: np.ndarray,
            Y: np.ndarray) -> 'RidgeRegressor':         
        if self.scale:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        if self.fit_intercept:
            self.intercept = Y.mean()
        else:
            self.intercept = 0
        X_transp_X = np.matmul(X.T, X)
        M = X_transp_X + self.lambd*np.eye(X_transp_X.shape[0])
        v = np.matmul(X.T, Y)
        self.beta = np.linalg.solve(M, v)
        return self
