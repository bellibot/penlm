import numpy as np
import copy

from sklearn.preprocessing import StandardScaler      
from sklearn.linear_model import Lasso, LogisticRegression
from base_estimators import BaseClassifier, BaseRegressor
from typing import Dict
              
                                                                          
class AdaptiveLassoClassifier(BaseClassifier):
    def __init__(self,
                 fit_intercept: bool = True,
                 class_weight: str = 'balanced',
                 scale: bool = True,
                 random_state: int = None,
                 solver: str = 'saga',
                 max_iter: int = 1000):
        self.class_weight = class_weight
        self.solver = solver 
        self.fit_intercept = fit_intercept
        self.parameters = {}
        self.lambd = None
        self.gamma = None
        self.beta_init = None
        self.beta = []
        self.intercept = []
        self.scale = scale
        self.scaler = None
        self.random_state = random_state
        self.max_iter = max_iter
        self.classes = None

    
    def set_parameters(self,
                       parameters: Dict):
        self.gamma = parameters['gamma']
        self.lambd = parameters['lambda']
        self.beta_init = parameters['beta_init'][1]
        self.parameters['gamma'] = self.gamma
        self.parameters['lambda'] = self.lambd
        self.parameters['beta_init'] = parameters['beta_init'][0] 
        if self.lambd <= 0:
            raise ValueError('Lambda <= 0')
        if self.gamma <= 0:
            raise ValueError('Gamma <= 0')
                            
                
    def fit(self,
            X: np.ndarray,
            Y: np.ndarray) -> 'AdaptiveLassoClassifier':
        if self.scale:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)            
        self.classes = np.unique(Y)
        n_classes = len(self.classes)
        if n_classes < 2:
            raise ValueError('''This estimator needs samples of at least 2 classes in the data, 
                                but the data contains only one class''')
        elif n_classes == 2:
            recipr_w = np.power(np.abs(self.beta_init[0]),
                                self.gamma)
            _X = X * np.tile(recipr_w, 
                             (X.shape[0], 1))
            estimator = LogisticRegression(penalty = 'l1',
                                           class_weight = self.class_weight,
                                           C = 1/self.lambd,
                                           fit_intercept = self.fit_intercept,
                                           random_state = self.random_state,
                                           solver = self.solver,
                                           max_iter = self.max_iter)
            estimator.fit(_X, Y)
            self.intercept.append(estimator.intercept_[0])
            self.beta.append(estimator.coef_[0]*recipr_w)
        else:    
            for j,k in enumerate(self.classes):
                recipr_w = np.power(np.abs(self.beta_init[j]),
                                    self.gamma)
                _X = X * np.tile(recipr_w, 
                                 (X.shape[0], 1))
                _Y = np.where(Y == k, 1, 0)     
                estimator = LogisticRegression(penalty = 'l1',
                                               class_weight = self.class_weight,
                                               C = 1/self.lambd,
                                               fit_intercept = self.fit_intercept,
                                               random_state = self.random_state,
                                               solver = self.solver,
                                               max_iter = self.max_iter)
                estimator.fit(_X, _Y)
                self.intercept.append(estimator.intercept_[0]) 
                self.beta.append(estimator.coef_[0]*recipr_w)
        self.intercept = np.array(self.intercept)
        self.beta = np.array(self.beta)
        return self


class AdaptiveLassoRegressor(BaseRegressor):
    def __init__(self,
                 fit_intercept: bool = True,
                 scale: bool = True,
                 random_state: int = None,
                 selection: str = 'cyclic',
                 max_iter: int = 1000):
        self.fit_intercept = fit_intercept
        self.parameters = {}
        self.lambd = None
        self.gamma = None
        self.beta_init = None
        self.beta = None
        self.intercept = None
        self.scale = scale
        self.scaler = None
        self.random_state = random_state
        self.selection = selection
        self.max_iter = max_iter

    
    def set_parameters(self,
                       parameters: Dict):
        self.gamma = parameters['gamma']
        self.lambd = parameters['lambda']
        self.beta_init = parameters['beta_init'][1]
        self.parameters['gamma'] = self.gamma
        self.parameters['lambda'] = self.lambd
        self.parameters['beta_init'] = parameters['beta_init'][0] 
        if self.lambd <= 0:
            raise ValueError('Lambda <= 0')
        if self.gamma <= 0:
            raise ValueError('Gamma <= 0')
                            
                
    def fit(self,
            X: np.ndarray,
            Y: np.ndarray) -> 'AdaptiveLassoRegressor':         
        if self.scale:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        recipr_w = np.power(np.abs(self.beta_init),
                            self.gamma)
        for i in range(X.shape[0]):
            X[i,:] = X[i,:]*recipr_w
        estimator = Lasso(alpha = self.lambd,
                          fit_intercept = self.fit_intercept,
                          random_state = self.random_state,
                          max_iter = self.max_iter,
                          selection = self.selection)
        estimator.fit(X, Y)
        self.intercept = estimator.intercept_ 
        self.beta = estimator.coef_*recipr_w
        return self
        
        
