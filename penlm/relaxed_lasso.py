import numpy as np

from sklearn.preprocessing import StandardScaler      
from sklearn.linear_model import Lasso, LogisticRegression
from base_estimators import BaseClassifier, BaseRegressor
from typing import Dict
             
                                                                          
class RelaxedLassoClassifier(BaseClassifier):
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
        self.phi = None
        self.beta = []
        self.intercept = []
        self.scale = scale
        self.scaler = None
        self.random_state = random_state
        self.max_iter = max_iter
        self.classes = None

    
    def set_parameters(self,
                       parameters: Dict):
        self.phi = parameters['phi']
        self.lambd = parameters['lambda']
        self.parameters['phi'] = self.phi
        self.parameters['lambda'] = self.lambd
        if self.lambd <= 0:
            raise ValueError('Lambda <= 0')
        if self.phi<=0 or self.phi>1:
            raise ValueError('Phi out of (0,1]')
        
          
    def fit(self,
            X: np.ndarray,
            Y: np.ndarray) -> 'RelaxedLassoClassifier':
        if self.scale:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)     
        self.classes = np.unique(Y)    
        n_classes = len(self.classes)
        if n_classes < 2:
            raise ValueError('''This estimator needs samples of at least 2 classes in the data, 
                                but the data contains only one class''')
        elif n_classes == 2:
            estimator = LogisticRegression(penalty = 'l1',
                                           class_weight = self.class_weight,
                                           C = 1/self.lambd,
                                           fit_intercept = self.fit_intercept,
                                           random_state = self.random_state,
                                           solver = self.solver,
                                           max_iter = self.max_iter)
            estimator.fit(X, Y)
            zero_indexes = np.where(np.isclose(estimator.coef_[0], 0))[0]
            _X = np.copy(X)
            _X[:,zero_indexes] = 0
            estimator = LogisticRegression(penalty = 'l1',
                                           class_weight = self.class_weight,
                                           C = 1/(self.lambd*self.phi),
                                           fit_intercept = self.fit_intercept,
                                           random_state = self.random_state,
                                           solver = self.solver,
                                           max_iter = self.max_iter)
            estimator.fit(_X, Y)
            self.intercept.append(estimator.intercept_[0]) 
            self.beta.append(estimator.coef_[0])
        else:    
            for j,k in enumerate(self.classes):
                _Y = np.where(Y == k, 1, 0)
                estimator = LogisticRegression(penalty = 'l1',
                                               class_weight = self.class_weight,
                                               C = 1/self.lambd,
                                               fit_intercept = self.fit_intercept,
                                               random_state = self.random_state,
                                               solver = self.solver,
                                               max_iter = self.max_iter)
                estimator.fit(X, _Y)
                zero_indexes = np.where(np.isclose(estimator.coef_[0], 0))[0]
                _X = np.copy(X)
                _X[:,zero_indexes] = 0
                estimator = LogisticRegression(penalty = 'l1',
                                               class_weight = self.class_weight,
                                               C = 1/(self.lambd*self.phi),
                                               fit_intercept = self.fit_intercept,
                                               random_state = self.random_state,
                                               solver = self.solver,
                                               max_iter = self.max_iter)
                estimator.fit(_X, _Y)
                self.intercept.append(estimator.intercept_[0]) 
                self.beta.append(estimator.coef_[0])
        self.intercept = np.array(self.intercept)
        self.beta = np.array(self.beta)            
        return self



class RelaxedLassoRegressor(BaseRegressor):
    def __init__(self,
                 fit_intercept: bool = True,
                 scale: bool = True,
                 random_state: int = None,
                 selection: str = 'cyclic',
                 max_iter: int = 1000):
        self.fit_intercept = fit_intercept
        self.parameters = {}
        self.lambd = None
        self.phi = None
        self.beta = None
        self.intercept = None
        self.estimator = None
        self.scale = scale
        self.scaler = None
        self.random_state = random_state
        self.max_iter = max_iter
        self.selection = selection

    
    def set_parameters(self,
                       parameters: Dict):
        self.phi = parameters['phi']
        self.lambd = parameters['lambda']
        self.parameters['phi'] = self.phi
        self.parameters['lambda'] = self.lambd
        if self.lambd <= 0:
            raise ValueError('Lambda <= 0')
        if self.phi<=0 or self.phi>1:
            raise ValueError('Phi out of (0,1]')
        
                
    def predict(self,
                X: np.ndarray) -> np.ndarray:
        if self.scale:
            X = self.scaler.transform(X)
        return self.estimator.predict(X)
                
                
    def fit(self,
            X: np.ndarray,
            Y: np.ndarray) -> 'RelaxedLassoRegressor':          
        if self.scale:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        estimator = Lasso(alpha = self.lambd,
                          fit_intercept = self.fit_intercept,
                          random_state = self.random_state,
                          selection = self.selection,
                          max_iter = self.max_iter)
        estimator.fit(X, Y)
        zero_indexes = np.where(np.isclose(estimator.coef_, 0))[0]
        X[:,zero_indexes] = 0
        estimator = Lasso(alpha = self.phi*self.lambd,
                          fit_intercept = self.fit_intercept,
                          random_state = self.random_state,
                          selection = self.selection,
                          max_iter = self.max_iter)
        self.estimator = estimator.fit(X, Y)
        self.intercept = estimator.intercept_ 
        self.beta = estimator.coef_
        return self
        
        
