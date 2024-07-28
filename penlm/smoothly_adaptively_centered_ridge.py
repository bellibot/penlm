import joblib
import numpy as np
import pyomo.environ as pyo

from pyomo.core.expr import numeric_expr
from pyomo.opt import SolverFactory
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from penlm.base_estimators import BaseClassifier, BaseRegressor
from abc import ABC, abstractmethod



class BaseSACR(ABC):
    def set_parameters(self,
                       parameters: dict):
        self.parameters['lambda'] = parameters['lambda']
        self.parameters['phi'] = parameters['phi']
        self.lambd = parameters['lambda']
        self.phi = parameters['phi']
        if self.lambd <= 0:
            raise ValueError('Lambda <= 0')
        if self.phi<=0 or self.phi>1:
            raise ValueError('Phi out of (0,1]')


    @abstractmethod
    def _init_ridge(self,
                    X: np.ndarray,
                    Y: np.ndarray):
        raise NotImplementedError()
            
                            
    def _joint_ridge(self,
                     X: np.ndarray,
                     Y: np.ndarray,
                     g: np.ndarray) -> tuple[int,np.ndarray,np.ndarray]:
        N = X.shape[0]
        P = X.shape[1]
        model = pyo.ConcreteModel()
        model.P = pyo.RangeSet(0, P-1)
        model.beta = pyo.Var(model.P, 
                             domain = pyo.Reals)
        model.w = pyo.Var(model.P, 
                          domain = pyo.NonNegativeReals)
        if self.fit_intercept:
            model.intercept = pyo.Var(domain = pyo.Reals,
                                      initialize = 0)
        for p in range(P):
            model.beta[p]= 0
            model.w[p]= 0
                    
        def constraint_w_integral(model):
            return (1/P)*sum(model.w[p] for p in model.P) == 1
        model.constraint_w_integral = pyo.Constraint(rule = constraint_w_integral)
        
        model.obj = pyo.Objective(expr = self._obj_expr(model, X, Y, g),
                                  sense = pyo.minimize)
        opt = SolverFactory(self.solver,
                            options={'max_iter': self.max_iter}) 
        opt.solve(model)
        beta = []
        w = []
        for p in range(P):
            beta.append(pyo.value(model.beta[p]))
            w.append(pyo.value(model.w[p]))   
        beta = np.array(beta)
        w = np.array(w)
        if self.fit_intercept:
            intercept = pyo.value(model.intercept)
        else:
            intercept = 0
        return intercept, beta, w
        

    @abstractmethod
    def _obj_expr(self,
                  model: pyo.ConcreteModel,
                  X: np.ndarray,
                  Y: np.ndarray,
                  g: np.ndarray) -> numeric_expr:
        raise NotImplementedError()


    def _get_first_penalty_expr(self,
                                model: pyo.ConcreteModel,
                                P: int,
                                g: np.ndarray) -> numeric_expr:
        return sum((model.beta[p] - g[p]*model.w[p])**2 for p in range(P))
    
    
    def _get_second_penalty_expr(self,
                                 model: pyo.ConcreteModel,
                                 P: int,
                                 g: np.ndarray) -> numeric_expr:
        return P*P*sum((g[p]*model.w[p] - 2*g[p+1]*model.w[p+1] + g[p+2]*model.w[p+2])**2 for p in range(P-2))
        
        
            
class SACRClassifier(BaseSACR, BaseClassifier):
    def __init__(self,
                 solver: str = 'ipopt',
                 scikit_solver: str = 'saga',
                 fit_intercept: bool = True,
                 max_iter: int = 100,
                 scikit_max_iter: int = 1000,
                 scale: bool = True,
                 class_weight: str = 'balanced'):
        self.solver = solver
        self.scikit_solver = scikit_solver
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.scikit_max_iter = scikit_max_iter
        self.scale = scale
        self.class_weight = class_weight
        self.parameters = {}
        self.beta = []
        self.w = []
        self.g = []
        self.intercept = []
        self.lambd = None
        self.phi = None
        self.scaler = None
        self.classes = None


    def fit(self,
            X: np.ndarray,
            Y: np.ndarray) -> 'SACRClassifier':
        if self.scale:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)            
        self.classes = np.unique(Y)
        n_classes = len(self.classes)
        if n_classes < 2:
            raise ValueError('''This estimator needs samples of at least 2 classes in the data, 
                                but the data contains only one class''')
        elif n_classes == 2:
            g = self._init_ridge(X, Y)
            intercept, beta, w = self._joint_ridge(X, Y, g)
            self.intercept.append(intercept)
            self.beta.append(beta)
            self.w.append(w)
            self.g.append(g)  
        else:
            for j,k in enumerate(self.classes):
                _Y = np.where(Y == k, 1, 0)
                g = self._init_ridge(X, _Y)
                intercept, beta, w = self._joint_ridge(X, _Y, g)
                self.intercept.append(intercept)
                self.beta.append(beta)
                self.w.append(w)
                self.g.append(g)
        self.intercept = np.array(self.intercept)
        self.beta = np.array(self.beta)
        self.w = np.array(self.w)
        self.g = np.array(self.g)            
        return self
        
        
    def _init_ridge(self,
                    X: np.ndarray,
                    Y: np.ndarray) -> np.ndarray:
        estimator = LogisticRegression(penalty = 'l2',
                                       C = 1/self.lambd,
                                       class_weight = self.class_weight,
                                       fit_intercept = self.fit_intercept,
                                       solver = self.scikit_solver,
                                       max_iter = self.scikit_max_iter)
        estimator.fit(X, Y)
        return estimator.coef_[0]


    def _obj_expr(self,
                  model: pyo.ConcreteModel,
                  X: np.ndarray,
                  Y: np.ndarray,
                  g: np.ndarray) -> numeric_expr:
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
        pen1 = self._get_first_penalty_expr(model, P, g)
        pen2 = self._get_second_penalty_expr(model, P, g)
        return obj + self.phi*self.lambd*pen1 + (1-self.phi)*self.lambd*pen2
                



class SACRRegressor(BaseSACR, BaseRegressor):
    def __init__(self,
                 solver: str = 'ipopt',
                 fit_intercept: bool = True,
                 max_iter: int = 100,
                 scikit_max_iter: int = 1000,
                 scale: bool = True):
        self.solver = solver
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.scikit_max_iter = scikit_max_iter
        self.scale = scale
        self.parameters = {}
        self.beta = None
        self.w = None
        self.g = None
        self.intercept = None
        self.lambd = None
        self.phi = None
        self.scaler = None


    def fit(self,
            X: np.ndarray,
            Y: np.ndarray) -> 'SACRRegressor':
        self.classes = np.unique(Y)    
        if self.scale:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        self.g = self._init_ridge(X, Y)
        self.intercept, self.beta, self.w = self._joint_ridge(X, Y, self.g)       
        return self
        
        
    def _init_ridge(self,
                    X: np.ndarray,
                    Y: np.ndarray) -> np.ndarray:
        estimator = Ridge(alpha = self.lambd,
                          fit_intercept = self.fit_intercept,
                          max_iter = self.scikit_max_iter)
        estimator.fit(X, Y)
        return estimator.coef_


    def _obj_expr(self,
                  model: pyo.ConcreteModel,
                  X: np.ndarray,
                  Y: np.ndarray,
                  g: np.ndarray) -> numeric_expr:
        N = X.shape[0]
        P = X.shape[1]
        if self.fit_intercept:
            obj = sum((Y[j]-sum(X[j,p]*model.beta[p] for p in range(P))-model.intercept)**2 for j in range(N))
        else:
            obj = sum((Y[j]-sum(X[j,p]*model.beta[p] for p in range(P)))**2 for j in range(N))
        pen1 = self._get_first_penalty_expr(model, P, g)
        pen2 = self._get_second_penalty_expr(model, P, g)
        return obj + self.phi*self.lambd*pen1 + (1-self.phi)*self.lambd*pen2



