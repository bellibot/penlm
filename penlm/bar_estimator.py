import numpy as np

from sklearn.preprocessing import StandardScaler      
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from base_estimators import BaseClassifier, BaseRegressor
from typing import Dict
                           
                                                                         
class BARClassifier(BaseClassifier):
    def __init__(self,
                 fit_intercept: bool = True,
                 max_iter: int = 5,
                 class_weight: str = 'balanced',
                 scale: bool = True,
                 random_state: int = None,
                 solver: str = 'saga',
                 scikit_max_iter: int = 1000,
                 scoring: str = 'balanced_accuracy'):
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
        self.n_iters = 0
        self.scikit_max_iter = scikit_max_iter
        self.scoring = scoring
        self.classes = None

    
    def set_parameters(self,parameters):
        self.lambd = parameters['lambda']
        self.parameters['lambda'] = self.lambd
        self.parameters['n_iters'] = 0
        if self.lambd <= 0:
            raise ValueError('Lambda <= 0')
                
                
    def fit(self,
            X: np.ndarray,
            Y: np.ndarray) -> 'BARClassifier':
        if self.scale:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)            
        self.classes = np.unique(Y)
        n_classes = len(self.classes)
        if n_classes < 2:
            raise ValueError('''This estimator needs samples of at least 2 classes in the data, 
                                but the data contains only one class''')
        elif n_classes == 2:
            estimator = LogisticRegression(penalty = 'l2',
                                           class_weight = self.class_weight,
                                           C = 1/self.lambd,
                                           fit_intercept = self.fit_intercept,
                                           random_state = self.random_state,
                                           solver = self.solver,
                                           max_iter = self.scikit_max_iter)
            estimator.fit(X, Y)
            recipr_w = np.square(estimator.coef_[0])
            _X = np.zeros(X.shape)
            for i in range(X.shape[0]):
                _X[i,:] = X[i,:]*recipr_w
            estimator.fit(_X, Y)
            best_intercept = estimator.intercept_[0] 
            best_beta = estimator.coef_[0]*recipr_w
            score = 1 - self._pred_score(X, Y, best_beta, best_intercept)
            score_decreased = True
            k = 0
            while (k < self.max_iter) and score_decreased:
                recipr_w = np.square(best_beta)
                _X = np.zeros(X.shape)
                for i in range(X.shape[0]):
                    _X[i,:] = X[i,:]*recipr_w
                estimator.fit(_X,Y)
                beta = estimator.coef_[0]*recipr_w
                intercept = estimator.intercept_[0]
                tmp_score = 1 - self._pred_score(X, Y, beta, intercept)
                if tmp_score > score:
                    score_decreased = False
                else:
                    best_beta = beta
                    best_intercept = intercept    
                    self.n_iters += 1
                    self.parameters['n_iters'] += 1
                    k += 1
            self.intercept.append(best_intercept)
            self.beta.append(best_beta) 
        else:    
            for j,k in enumerate(self.classes):            
                estimator = LogisticRegression(penalty = 'l2',
                                               class_weight = self.class_weight,
                                               C = 1/self.lambd,
                                               fit_intercept = self.fit_intercept,
                                               random_state = self.random_state,
                                               solver = self.solver,
                                               max_iter = self.scikit_max_iter)
                _Y = np.where(Y == k, 1, 0)                                    
                estimator.fit(X, _Y)
                recipr_w = np.square(estimator.coef_[0])
                _X = np.zeros(X.shape)
                for i in range(X.shape[0]):
                    _X[i,:] = X[i,:]*recipr_w
                estimator.fit(_X, _Y)
                best_intercept = estimator.intercept_[0] 
                best_beta = estimator.coef_[0]*recipr_w
                score = 1 - self._pred_score(X, _Y, best_beta, best_intercept)
                score_decreased = True
                k = 0
                while (k < self.max_iter) and score_decreased:
                    recipr_w = np.square(best_beta)
                    _X = np.zeros(X.shape)
                    for i in range(X.shape[0]):
                        _X[i,:] = X[i,:]*recipr_w
                    estimator.fit(_X, _Y)
                    beta = estimator.coef_[0]*recipr_w
                    intercept = estimator.intercept_[0]
                    tmp_score = 1 - self._pred_score(X, _Y, beta, intercept)
                    if tmp_score > score:
                        score_decreased = False
                    else:
                        best_beta = beta
                        best_intercept = intercept    
                        self.n_iters += 1
                        self.parameters['n_iters'] += 1
                        k += 1
                self.intercept.append(best_intercept)
                self.beta.append(best_beta)
        self.intercept = np.array(self.intercept)
        self.beta = np.array(self.beta)            
        return self


    def _pred_score(self, 
                    X: np.ndarray, 
                    Y: np.ndarray, 
                    beta: np.ndarray, 
                    intercept: int) -> np.ndarray:
        pred_Y = []
        for x in X:
            if np.dot(beta,x) + intercept > 0:
                pred_Y.append(1)
            else:
                pred_Y.append(0)
        if self.scoring == 'balanced_accuracy':
            score = balanced_accuracy_score(Y,pred_Y) 
        elif self.scoring == 'accuracy':
            score = accuracy_score(Y,pred_Y) 
        else:
            raise ValueError(f'Illegal scoring {self.scoring}')
        return score 
        


class BARRegressor(BaseRegressor):
    def __init__(self,
                 fit_intercept: bool = True,
                 max_iter: int = 5,
                 scale: bool = True,
                 random_state: int = None,
                 scikit_max_iter: int = 1000,
                 scoring: str = 'mse'):
        self.fit_intercept = fit_intercept
        self.parameters = {}
        self.lambd = None
        self.phi = None
        self.beta = None
        self.intercept = None
        self.scale = scale
        self.scaler = None
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_iters = 0
        self.scikit_max_iter = scikit_max_iter
        self.scoring = scoring

    
    def set_parameters(self,parameters):
        self.lambd = parameters['lambda']
        self.parameters['lambda'] = self.lambd
        self.parameters['n_iters'] = 0
        if self.lambd <= 0:
            raise ValueError('Lambda <= 0')
                
                
    def fit(self,
            X: np.ndarray,
            Y: np.ndarray) -> 'BARestimator':      
        if self.scale:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        estimator = Ridge(alpha = self.lambd,
                          fit_intercept = self.fit_intercept,
                          random_state = self.random_state,
                          max_iter = self.scikit_max_iter)
        estimator.fit(X, Y)
        recipr_w = np.square(estimator.coef_)
        _X = np.zeros(X.shape)
        for i in range(X.shape[0]):
            _X[i,:] = X[i,:]*recipr_w
        estimator.fit(_X, Y)
        self.intercept = estimator.intercept_ 
        self.beta = estimator.coef_*recipr_w
        score = self._pred_score(X, Y, self.beta, self.intercept)
        score_decreased = True
        k = 0
        while (k < self.max_iter) and score_decreased:
            recipr_w = np.square(self.beta)
            _X = np.zeros(X.shape)
            for i in range(X.shape[0]):
                _X[i,:] = X[i,:]*recipr_w
            estimator.fit(_X, Y)
            beta = estimator.coef_*recipr_w
            intercept = estimator.intercept_
            tmp_score = self._pred_score(X, Y, beta, intercept)
            if tmp_score > score:
                score_decreased = False
            else:
                self.beta = beta
                self.intercept = intercept    
                self.n_iters += 1
                self.parameters['n_iters'] += 1
                k += 1
        return self


    def _pred_score(self, 
                    X: np.ndarray, 
                    Y: np.ndarray, 
                    beta: np.ndarray, 
                    intercept: int) -> np.ndarray:
        pred_Y = []
        for x in X:
            pred = np.dot(beta,x) + intercept
            pred_Y.append(pred)
        pred_Y = np.array(pred_Y)
        if self.scoring == 'mse':
            score = np.mean((pred_Y-Y)**2)
        elif self.scoring == 'mae':  
            score = np.mean(np.abs(pred_Y-Y))
        else:
            raise ValueError(f'Illegal scoring {self.scoring}')
        return score 
        
                
