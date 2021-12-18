import numpy as np
import pyomo.environ as pyo

from pyomo.core.expr import numeric_expr
from pyomo.opt import SolverFactory
from sklearn.preprocessing import StandardScaler
from penlm.base_estimators import BaseClassifier, BaseRegressor
from abc import ABC, abstractmethod
from typing import Dict, Tuple
                          

class BaseNNGarrote(ABC):
    def set_parameters(self,
                       parameters: Dict):
        self.lambd = parameters['lambda']
        self.beta_init = parameters['beta_init'][1]
        self.parameters['lambda'] = self.lambd
        self.parameters['beta_init'] = parameters['beta_init'][0] 
        if self.lambd <= 0:
            raise ValueError('Lambda <= 0')

        
    def _solve_optim(self,
                     X: np.ndarray,
                     Y: np.ndarray,
                     beta_init: np.ndarray) -> Tuple[int,np.ndarray,np.ndarray]:
        P = X.shape[1]
        N = X.shape[0]
        for i in range(N):
            X[i,:] = X[i,:]*beta_init 
        model = pyo.ConcreteModel()
        model.P = pyo.RangeSet(0, P-1)
        model.c = pyo.Var(model.P, 
                          domain = pyo.NonNegativeReals)
        if self.fit_intercept:
            model.intercept = pyo.Var(domain = pyo.Reals,
                                      initialize = 0)
        for p in range(P):
            model.c[p]= 0
            
        model.obj = pyo.Objective(expr = self._obj_expr(model,X,Y), 
                                  sense = pyo.minimize)        
        opt = SolverFactory(self.solver,
                            options = {'max_iter': self.max_iter}) 
        opt.solve(model)
        c = []
        for p in range(P):
            c.append(pyo.value(model.c[p]))   
        beta = np.array(c)*beta_init
        if self.fit_intercept:
            intercept = pyo.value(model.intercept)
        else:
            intercept = 0
        return intercept, beta, c
        

    @abstractmethod
    def _obj_expr(self,
                  model: pyo.ConcreteModel,
                  X: np.ndarray,
                  Y: np.ndarray) -> numeric_expr:
        pass
        
        
                                                                                  
class NNGarroteClassifier(BaseNNGarrote, BaseClassifier):
    def __init__(self,
                 solver: str = 'ipopt',
                 fit_intercept: bool = True,
                 scale: bool = True,
                 max_iter: int = 100,
                 class_weight: str = 'balanced'):    
        self.solver = solver
        self.fit_intercept = fit_intercept
        self.class_weight = class_weight
        self.parameters = {}
        self.lambd = None
        self.max_iter = max_iter
        self.beta_init = None
        self.beta = []
        self.c = []
        self.intercept = []
        self.scale = scale
        self.scaler = None
        self.classes = None


    def fit(self,
            X: np.ndarray,
            Y: np.ndarray) -> 'NNGarroteClassifier':
        if self.scale:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)            
        self.classes = np.unique(Y)
        n_classes = len(self.classes)
        if n_classes < 2:
            raise ValueError('''This estimator needs samples of at least 2 classes in the data, 
                                but the data contains only one class''')
        elif n_classes == 2:
            intercept, beta, c = self._solve_optim(X, Y, self.beta_init[0])
            self.intercept.append(intercept)
            self.beta.append(beta)
            self.c.append(c)
        else:
            for j,k in enumerate(self.classes):
                _Y = np.where(Y == k, 1, 0) 
                intercept, beta, c = self._solve_optim(X, _Y, self.beta_init[j]) 
                self.intercept.append(intercept)
                self.beta.append(beta)
                self.c.append(c)
        self.intercept = np.array(self.intercept)
        self.beta = np.array(self.beta) 
        self.c = np.array(self.c)                        
        return self
        
        
    def _obj_expr(self,
                  model: pyo.ConcreteModel,
                  X: np.ndarray,
                  Y: np.ndarray) -> numeric_expr:
        N = X.shape[0]
        P = X.shape[1]
        if self.class_weight=='balanced': 
            unique_weights = N/(len(np.unique(Y))*np.bincount(Y))
            weights = np.zeros(N)
            for i in np.unique(Y):
                weights[np.where(Y==i)] = unique_weights[i]
        else:
            weights = np.ones(N)
        if self.fit_intercept:
            obj = -sum(weights[j]*(Y[j]*(sum(X[j,p]*model.c[p] for p in range(P))+model.intercept) - pyo.log(1+pyo.exp(model.intercept+sum(X[j,p]*model.c[p] for p in range(P)))))  for j in range(N))
        else:
            obj = -sum(weights[j]*(Y[j]*(sum(X[j,p]*model.c[p] for p in range(P))) - pyo.log(1+pyo.exp(sum(X[j,p]*model.c[p] for p in range(P)))))  for j in range(N))
        
        pen = sum(model.c[p] for p in range(P))
        return obj + self.lambd*pen
        


class NNGarroteRegressor(BaseNNGarrote, BaseRegressor):
    def __init__(self,
                 solver: str = 'ipopt',
                 fit_intercept: bool = True,
                 scale: bool = True,
                 max_iter: int = 100):    
        self.solver = solver
        self.fit_intercept = fit_intercept
        self.parameters = {}
        self.lambd = None
        self.max_iter = max_iter
        self.beta_init = None
        self.beta = None
        self.c = None
        self.intercept = None
        self.scale = scale
        self.scaler = None


    def fit(self,
            X: np.ndarray,
            Y: np.ndarray) -> 'NNGarroteRegressor':
        if self.scale:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        self.intercept, self.beta, self.c = self._solve_optim(X, Y, self.beta_init)
        return self
        
        
    def _obj_expr(self,
                  model: pyo.ConcreteModel,
                  X: np.ndarray,
                  Y: np.ndarray) -> numeric_expr:
        N = X.shape[0]
        P = X.shape[1]
        if self.fit_intercept:
            obj = sum((Y[j]-sum(X[j,p]*model.c[p] for p in range(P))-model.intercept)**2 for j in range(N))
        else:
            obj = sum((Y[j]-sum(X[j,p]*model.c[p] for p in range(P)))**2 for j in range(N))
        
        pen = sum(model.c[p] for p in range(P))
        return obj + self.lambd*pen
        

