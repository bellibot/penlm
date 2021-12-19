import numpy as np

from typing import Dict, Callable
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, r2_score
                                  

class BaseClassifier(ABC):
    estimator_type = 'classifier'
    
    
    def predict(self,
                X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
        
        
    def predict_proba(self,
                      X: np.ndarray) -> np.ndarray:           
        if self.scale:
            X = self.scaler.transform(X)
        intercepts = np.tile(self.intercept, 
                             (X.shape[0], 1))                   
        log_odds = np.matmul(X, self.beta.transpose()) + intercepts
        probs_one = (1 / (1 + np.exp(-log_odds)))        
        if len(self.classes) == 2:
            probs = np.vstack([1 - probs_one[:,0], probs_one[:,0]]).T
        else:
            probs = probs_one/probs_one.sum(axis=1).reshape((probs_one.shape[0], -1))
        return probs
                
                
    def score(self,
              X: np.ndarray,
              Y: np.ndarray,
              scoring: Callable = None) -> np.ndarray:
        pred_Y = self.predict(X) 
        if scoring == None:
            score = accuracy_score(Y, pred_Y) 
        elif callable(scoring):
            score = scoring(Y, pred_Y)
        else:
            raise ValueError(f'Illegal scoring {scoring}')          
        return score
        
        
    @abstractmethod
    def set_parameters(self,
                       parameters: Dict):
        pass
          
                
    @abstractmethod            
    def fit(self,
            X: np.ndarray,
            Y: np.ndarray) -> 'BaseClassifier':          
        pass    
        
        
        
class BaseRegressor(ABC):
    estimator_type = 'regressor'
    
    
    def predict(self,
                X: np.ndarray) -> np.ndarray:
        if self.scale:
            X = self.scaler.transform(X)
        pred_Y = []
        for x in X:
            pred = np.dot(self.beta,x) + self.intercept
            pred_Y.append(pred)
        return np.array(pred_Y)
        
            
    def score(self,
              X: np.ndarray,
              Y: np.ndarray,
              scoring: Callable = None) -> np.ndarray:
        pred_Y = self.predict(X) 
        if scoring == None:
            score = r2_score(Y, pred_Y) 
        elif callable(scoring):
            score = scoring(Y, pred_Y)
        else:
            raise ValueError(f'Illegal scoring {scoring}')          
        return score
        
        
    @abstractmethod
    def set_parameters(self,
                       parameters: Dict):
        pass
          
                
    @abstractmethod            
    def fit(self,
            X: np.ndarray,
            Y: np.ndarray) -> 'BaseRegressor':          
        pass    




