import numpy as np
import itertools, copy, joblib

from typing import Dict
               
                           
class GridSearchCV:
    def __init__(self,
                 estimator: 'BaseEstimator',
                 parameters: Dict,
                 cv: 'ScikitCVObj',
                 scoring: str = None,
                 verbose: bool = False,
                 n_jobs: int = -1):
        self.parameters_packed = parameters
        flattened = [[(key, val) for val in values] for key, values in self.parameters_packed.items()]
        self.parameters_unpacked = [dict(items) for items in itertools.product(*flattened)]
        self.estimator = estimator
        self.cv = cv
        if scoring == None:
            if estimator.estimator_type=='classifier':
                self.scoring = 'balanced_accuracy'
            elif estimator.estimator_type=='regressor':
                self.scoring = 'r2'
        else:
            self.scoring = scoring
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.fold_estimators = []
        self.fold_best_estimators = []
        self.fold_train_scores = []
        self.fold_test_scores = []
        self.mean_fold_train_scores = []
        self.mean_fold_test_scores = []
        self.std_fold_train_scores = []
        self.std_fold_test_scores = []
        self.best_estimator = None
        self.best_parameters = None
        self.best_estimator_train_score = None 


    def fit(self, 
            X: np.ndarray, 
            Y: np.ndarray):
        for counter,(train_index, test_index) in enumerate(self.cv.split(X,Y)):
            estimators = []
            for param_dict in self.parameters_unpacked:
                estimator = copy.deepcopy(self.estimator)
                estimator.set_parameters(param_dict) 
                estimators.append(estimator)
            estimators = joblib.Parallel(n_jobs=self.n_jobs)(joblib.delayed(estimators[i].fit)(X[train_index],Y[train_index]) for i in range(len(self.parameters_unpacked)))
            inner_train_scores = joblib.Parallel(n_jobs=self.n_jobs)(joblib.delayed(estimators[i].score)(X[train_index],Y[train_index],self.scoring) for i in range(len(self.parameters_unpacked)))
            inner_test_scores = joblib.Parallel(n_jobs=self.n_jobs)(joblib.delayed(estimators[i].score)(X[test_index],Y[test_index],self.scoring) for i in range(len(self.parameters_unpacked)))
            self.fold_estimators.append(estimators)
            self.fold_train_scores.append(inner_train_scores)
            self.fold_test_scores.append(inner_test_scores)
            best_index = inner_test_scores.index(max(inner_test_scores))
            self.fold_best_estimators.append(estimators[best_index])
            if self.verbose:
                print()
                print(' Grid Search Iternum: {}'.format(counter))
                print()
                for p,s in zip(self.parameters_unpacked,inner_test_scores):   
                    print(' Inner Parameters: {}'.format(p))
                    print(' Inner Score: {}'.format(s))
                    print()
                print()
                if len(set(inner_test_scores))==1:
                    print(' All the parameter tuples have the same score in this fold')  
        self.mean_fold_train_scores = np.mean(self.fold_train_scores, axis=0)
        self.mean_fold_test_scores = np.mean(self.fold_test_scores, axis=0)
        self.std_fold_train_scores = np.std(self.fold_train_scores, axis=0)
        self.std_fold_test_scores = np.std(self.fold_test_scores, axis=0)
        best_index = np.argmax(self.mean_fold_test_scores)
        self.best_parameters = self.parameters_unpacked[best_index]
        if self.verbose:
            print()
            print(' Best Parameters: {}'.format(self.best_parameters))
            print()     
        estimator = copy.deepcopy(self.estimator)
        estimator.set_parameters(self.best_parameters)           
        estimator.fit(X,Y)
        self.best_estimator = estimator
        self.best_estimator_train_score = estimator.score(X,Y,self.scoring)
        
        
