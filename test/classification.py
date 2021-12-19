import numpy as np
import penlm.grid_search as gs
from penlm.smooth_linear_model import SmoothLinearClassifier
from penlm.smoothly_adaptively_centered_ridge import SACRClassifier
from penlm.relaxed_lasso import RelaxedLassoClassifier
from penlm.adaptive_lasso import AdaptiveLassoClassifier
from penlm.bar_estimator import BARClassifier
from penlm.non_negative_garrote import NNGarroteClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.metrics import balanced_accuracy_score
            
            
if __name__ == "__main__":
    import warnings, os
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
    
    random_state = 46
    X, Y = make_classification(n_samples = 90,
                               n_features = 120,
                               n_redundant = 50,
                               n_repeated = 0,
                               n_informative = 50,
                               n_classes = 3,
                               random_state = random_state)          
    pyomo_solver = 'ipopt'
    fit_intercept = True
    scoring = None
    #scoring = balanced_accuracy_score
    scale = True
    n_splits = 2
    n_splits_grid_search = 2
    begin = -3 
    end = 3 
    n_lambda = 5     
    lambda_list = np.logspace(begin,
                              end,
                              num = n_lambda,
                              base = 2)
    phi_list_sacr = np.linspace(0,1,5)[1:]
    gamma_list_adaptive_lasso = [0.001,0.01,0.1,1]
    phi_list_relaxo = phi_list_sacr

    ridge_score_list = []
    roughness_score_list = []
    sacr_score_list = []
    garrote_score_list = []
    relaxed_lasso_score_list = [] 
    adaptive_lasso_score_list = [] 
    bar_score_list = []  
           
    ### Train/Test splitter ###
    cv = StratifiedKFold(n_splits = n_splits,
                         random_state = random_state,
                         shuffle = True)
                   
    for counter,(train_index,test_index) in enumerate(cv.split(X,Y)):
        ### Train/Validation splitter (for Grid Search) ###
        _cv = StratifiedKFold(n_splits = n_splits_grid_search, 
                              random_state = random_state,
                              shuffle = True)
       
        ### Ridge ###    
        estimator = LogisticRegression(penalty = 'l2',
                                       multi_class = 'ovr',
                                       class_weight = 'balanced',
                                       fit_intercept = fit_intercept,
                                       solver = 'saga',
                                       max_iter = 1000)
        parameters = {'ridge__C':lambda_list}
        scaler = StandardScaler()   
        pipeline = Pipeline(steps = [('scaler',scaler),
                                     ('ridge', estimator)])    
        grid_search = GridSearchCV(pipeline,
                                   parameters,
                                   cv = _cv,
                                   n_jobs = -1)
        grid_search.fit(X[train_index],Y[train_index])
        beta_ridge = grid_search.best_estimator_[1].coef_
        score = grid_search.score(X[test_index],
                                  Y[test_index])
        ridge_score_list.append(score)
        
                
        ### ROUGHNESS ###
        estimator = SmoothLinearClassifier(pyomo_solver,
                                           fit_intercept = fit_intercept,
                                           scale = scale,
                                           penalty_type = 'd2')
        parameters = {'lambda':lambda_list}       
        grid_search = gs.GridSearchCV(estimator,
                                      parameters,
                                      _cv,
                                      scoring = scoring)
        grid_search.fit(X[train_index],
                        Y[train_index])
        score = grid_search.score(X[test_index],
                                  Y[test_index])
        roughness_score_list.append(score)


        ### Relaxed Lasso ###
        estimator = RelaxedLassoClassifier(fit_intercept = fit_intercept,
                                           random_state = random_state,
                                           scale = scale)
        parameters = {'lambda':lambda_list, 
                      'phi':phi_list_relaxo}
         
        grid_search = gs.GridSearchCV(estimator,
                                      parameters,
                                      _cv,
                                      scoring = scoring)
        grid_search.fit(X[train_index],
                        Y[train_index])
        score = grid_search.score(X[test_index],
                                  Y[test_index])
        relaxed_lasso_score_list.append(score)


        ### Adaptive Lasso ###
        estimator = AdaptiveLassoClassifier(fit_intercept = fit_intercept,
                                            random_state = random_state,
                                            scale = scale)
        parameters = {'beta_init':[('ridge',beta_ridge)],
                      'lambda':lambda_list, 
                      'gamma':gamma_list_adaptive_lasso}       
        grid_search = gs.GridSearchCV(estimator,
                                      parameters,
                                      _cv,
                                      scoring = scoring)
        grid_search.fit(X[train_index],
                        Y[train_index])
        score = grid_search.score(X[test_index],
                                  Y[test_index])
        adaptive_lasso_score_list.append(score)
        
        
        ### BAR estimator ###
        estimator = BARClassifier(fit_intercept = fit_intercept,
                                  random_state = random_state,
                                  scale = scale,
                                  scoring = scoring)
        parameters = {'lambda':lambda_list}       
        grid_search = gs.GridSearchCV(estimator,
                                      parameters,
                                      _cv,
                                      scoring = scoring)
        grid_search.fit(X[train_index],
                        Y[train_index])
        score = grid_search.score(X[test_index],
                                  Y[test_index])
        bar_score_list.append(score)
            
        
        ### NN garrote ###
        estimator = NNGarroteClassifier(pyomo_solver,
                                        fit_intercept = fit_intercept,
                                        scale = scale)
        parameters = {'beta_init':[('ridge',beta_ridge)], 
                      'lambda':lambda_list}       
        grid_search = gs.GridSearchCV(estimator,
                                      parameters,
                                      _cv,
                                      scoring = scoring)
        grid_search.fit(X[train_index],
                        Y[train_index])
        score = grid_search.score(X[test_index],
                                  Y[test_index])
        garrote_score_list.append(score)   
                        
        ### SACR ###
        estimator = SACRClassifier(solver = pyomo_solver,
                                   fit_intercept = fit_intercept,
                                   scale = scale)
        parameters = {'phi':phi_list_sacr,'lambda':lambda_list}
        grid_search = gs.GridSearchCV(estimator,
                                      parameters,
                                      _cv,
                                      scoring = scoring)
        grid_search.fit(X[train_index],
                        Y[train_index])
        score = grid_search.score(X[test_index],
                                  Y[test_index])
        sacr_score_list.append(score)
        
    print(f'RIDGE TEST SCORE:          {np.mean(ridge_score_list)}')
    print(f'ROUGHNESS TEST SCORE:      {np.mean(roughness_score_list)}')
    print(f'BAR TEST SCORE:            {np.mean(bar_score_list)}')
    print(f'RELAXED LASSO TEST SCORE:  {np.mean(relaxed_lasso_score_list)}')
    print(f'ADAPTIVE LASSO TEST SCORE: {np.mean(adaptive_lasso_score_list)}')
    print(f'GARROTE TEST SCORE:        {np.mean(garrote_score_list)}')
    print(f'SACR TEST SCORE:           {np.mean(sacr_score_list)}')
        

