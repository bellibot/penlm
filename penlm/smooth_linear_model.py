import numpy as np
import pyomo.environ as pyo

from pyomo.core.expr import numeric_expr
from pyomo.opt import SolverFactory
from sklearn.preprocessing import StandardScaler
from penlm.base_estimators import BaseClassifier, BaseRegressor
from abc import ABC, abstractmethod
from typing import Dict, Tuple


class BaseSmoothLinear(ABC):
    penalty_types = ['d1','d2']  

    
    def set_parameters(self,
                       parameters: Dict):
        self.lambd = parameters['lambda']
        self.parameters['lambda'] = self.lambd
        if self.lambd <= 0:
            raise ValueError('Lambda <= 0')


    def _solve_optim(self,
                     X: np.ndarray,
                     Y: np.ndarray) -> Tuple[int, np.ndarray]:
        P = X.shape[1]
        N = X.shape[0]
        model = pyo.ConcreteModel()
        model.P = pyo.RangeSet(0, P-1)
        model.beta = pyo.Var(model.P, 
                             domain = pyo.Reals)
        if self.fit_intercept:
            model.intercept = pyo.Var(domain = pyo.Reals,
                                      initialize = 0)
        for p in range(P):
            model.beta[p]= 0
        
        model.obj = pyo.Objective(expr = self._obj_expr(model,X,Y), 
                                  sense = pyo.minimize)
        opt = SolverFactory(self.solver,
                            options = {'max_iter': self.max_iter}) 
        opt.solve(model)
        
        beta = []
        for p in range(P):
            beta.append(pyo.value(model.beta[p]))   
        beta = np.array(beta)
        if self.fit_intercept:
            intercept = pyo.value(model.intercept)
        else:
            intercept = 0
        return intercept, beta


    @abstractmethod
    def _obj_expr(self,
                  model: pyo.ConcreteModel,
                  X: np.ndarray,
                  Y: np.ndarray) -> numeric_expr:
        pass
       
        
    def _get_penalty_expr(self,
                          model: pyo.ConcreteModel,
                          P: int) -> numeric_expr:
        pen = None
        if self.penalty_type=='d2':
            pen = P*P*sum((model.beta[p] - 2*model.beta[p+1] + model.beta[p+2])**2 for p in range(P-2))
        elif self.penalty_type=='d1':
            pen = P*sum((model.beta[p] - model.beta[p+1])**2 for p in range(P-1))                
        return pen 
        
        
                                                                                  
class SmoothLinearClassifier(BaseSmoothLinear, BaseClassifier):
    def __init__(self,
                 solver: str = 'ipopt',
                 penalty_type: str = 'd2',
                 fit_intercept: bool = True,
                 scale: bool = True,
                 max_iter: int = 100,
                 class_weight: str = 'balanced'):
        if penalty_type in self.penalty_types:
            self.penalty_type = penalty_type
        else:
            raise ValueError(f'Illegal Penalization Type == {penalty_type}')     
        self.solver = solver
        self.fit_intercept = fit_intercept
        self.class_weight = class_weight
        self.parameters = {}
        self.lambd = None
        self.max_iter = max_iter
        self.beta = []
        self.intercept = []
        self.scale = scale
        self.scaler = None
        self.classes = None


    def fit(self,
            X: np.ndarray,
            Y: np.ndarray) -> 'SmoothLinearClassifier': 
        if self.scale:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)            
        self.classes = np.unique(Y)
        n_classes = len(self.classes)
        if n_classes < 2:
            raise ValueError('''This estimator needs samples of at least 2 classes in the data, 
                                but the data contains only one class''')
        elif n_classes == 2:
            intercept, beta = self._solve_optim(X, Y)
            self.intercept.append(intercept)
            self.beta.append(beta)
        else:
            for j,k in enumerate(self.classes):
                _Y = np.where(Y == k, 1, 0)
                intercept, beta = self._solve_optim(X, _Y)
                self.intercept.append(intercept)
                self.beta.append(beta)
        self.intercept = np.array(self.intercept)
        self.beta = np.array(self.beta)    
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
            obj = -sum(weights[j]*(Y[j]*(sum(X[j,p]*model.beta[p] for p in range(P))+model.intercept) - pyo.log(1+pyo.exp(model.intercept+sum(X[j,p]*model.beta[p] for p in range(P)))))  for j in range(N))
        else:
            obj = -sum(weights[j]*(Y[j]*(sum(X[j,p]*model.beta[p] for p in range(P))) - pyo.log(1+pyo.exp(sum(X[j,p]*model.beta[p] for p in range(P)))))  for j in range(N))
        pen = self._get_penalty_expr(model, P)
        return obj + self.lambd*pen      
                
 
 
class SmoothLinearRegressor(BaseSmoothLinear, BaseRegressor):
    def __init__(self,
                 solver: str = 'ipopt',
                 penalty_type: str = 'd2',
                 fit_intercept: bool = True,
                 scale: bool = True,
                 max_iter: int = 100):
        if penalty_type in self.penalty_types:
            self.penalty_type = penalty_type
        else:
            raise ValueError(f'Illegal Penalization Type == {penalty_type}')        
        self.solver = solver
        self.fit_intercept = fit_intercept
        self.parameters = {}
        self.lambd = None
        self.max_iter = max_iter
        self.beta = None
        self.intercept = None
        self.scale = scale
        self.scaler = None


    def fit(self,
            X: np.ndarray,
            Y: np.ndarray) -> 'SmoothLinearRegressor':    
        if self.scale:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        self.intercept, self.beta = self._solve_optim(X, Y)
        return self
        
        
    def _obj_expr(self,
                  model: pyo.ConcreteModel,
                  X: np.ndarray,
                  Y: np.ndarray) -> numeric_expr:
        N = X.shape[0]
        P = X.shape[1]
        if self.fit_intercept:
            obj = sum((Y[j]-sum(X[j,p]*model.beta[p] for p in range(P))-model.intercept)**2 for j in range(N))
        else:
            obj = sum((Y[j]-sum(X[j,p]*model.beta[p] for p in range(P)))**2 for j in range(N))
        pen = self._get_penalty_expr(model, P)
        return obj + self.lambd*pen
        

