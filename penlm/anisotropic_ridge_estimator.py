import numpy as np

from sklearn.preprocessing import StandardScaler
from penlm.base_estimators import BaseRegressor


class AnisotropicRidgeRegressor(BaseRegressor):
    def __init__(self,
                 fit_intercept: bool = True,
                 scale: bool = True):
        self.fit_intercept = fit_intercept
        self.parameters = {}
        self.lambd = None
        self.beta_ridge = None
        self.beta = None
        self.intercept = None
        self.scale = scale
        self.scaler = None
        self.Q = None
        self.W = None
        self.A = None
        self.eigenvalues_XX = None
        self.eigenvalues_W = None
        self.eigenvalues_A = None

    
    def set_parameters(self,
                       parameters: dict):
        self.lambd = parameters['lambda']
        self.parameters['lambda'] = self.lambd


    def fit(self,
            X: np.ndarray,
            Y: np.ndarray) -> 'AnisotropicRidgeRegressor':         
        if self.scale:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        if self.fit_intercept:
            self.intercept = Y.mean()
        else:
            self.intercept = 0
        X_transp_X = np.matmul(X.T, X)
        self._compute_ridge_solution(X.T,
                                     X_transp_X,
                                     Y)
        self._compute_eigenvalues_W(X_transp_X)
        self._compute_W()
        self._compute_A()
        self.beta = np.matmul(self.A,
                              self.beta_ridge)
        return self


    def _compute_ridge_solution(self,
                                X_transp,
                                X_transp_X,
                                Y):
        M = X_transp_X + self.lambd*np.eye(X_transp_X.shape[0])
        v = np.matmul(X_transp,Y)
        self.beta_ridge = np.linalg.solve(M,v)


    def _compute_eigenvalues_W(self,
                               X_transp_X):
        self.eigenvalues_XX, self.Q = np.linalg.eigh(X_transp_X)
        if self.lambd > 0:
            self.eigenvalues_W = -(self.eigenvalues_XX + self.lambd)
        else:
            self.eigenvalues_W = np.empty_like(self.eigenvalues_XX)
            indexes = np.where(self.lambd == -self.eigenvalues_XX)[0]
            self.eigenvalues_W[indexes] = 0
            indexes = np.where(self.lambd < -self.eigenvalues_XX)[0]
            self.eigenvalues_W[indexes] = -(self.eigenvalues_XX[indexes] + self.lambd)
            indexes = np.where(self.lambd > -self.eigenvalues_XX)[0]
            w = self.lambd*(self.eigenvalues_XX + self.lambd)
            w = np.divide(w,
                          self.eigenvalues_XX,
                          out=np.zeros_like(w),
                          where=self.eigenvalues_XX!=0)
            self.eigenvalues_W[indexes] = w[indexes]


    def _compute_W(self):
        self.W = np.zeros(self.Q.shape)
        np.fill_diagonal(self.W,
                         self.eigenvalues_W)
        self.W = np.matmul(self.Q,
                           self.W)
        self.W = np.matmul(self.W,
                           self.Q.T)


    def _compute_A(self):
        tmp = self.eigenvalues_XX+self.lambd
        tmp = np.divide(self.eigenvalues_W,
                        tmp,
                        out=np.zeros_like(tmp),
                        where=tmp!=0)
        self.eigenvalues_A = 1 + tmp
        self.A = np.zeros_like(self.Q)
        np.fill_diagonal(self.A,
                         self.eigenvalues_A)
        self.A = np.matmul(self.Q,
                           self.A)
        self.A = np.matmul(self.A,
                           self.Q.T)

