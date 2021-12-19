import numpy as np
import penlm.grid_search as gs
from penlm.smooth_linear_model import SmoothLinearRegressor
from penlm.smoothly_adaptively_centered_ridge import SACRRegressor 
from penlm.relaxed_lasso import RelaxedLassoRegressor
from penlm.adaptive_lasso import AdaptiveLassoRegressor
from penlm.bar_estimator import BARRegressor
from penlm.non_negative_garrote import NNGarroteRegressor 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_regression
from sklearn.metrics import explained_variance_score

            
if __name__ == "__main__":
    import warnings, os
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

    random_state = 46
    X, Y = make_regression(n_samples = 90,
                           n_features = 120,
                           n_informative = 80,
                           noise = 1,
                           shuffle = True,
                           random_state = random_state)                                      
    pyomo_solver = 'ipopt'
    fit_intercept = True
    scoring = None
    #scoring = explained_variance_score
    scale = True
    n_splits = 2
    n_splits_grid_search = 2
    begin = -5 
    end = 5
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
    cv = KFold(n_splits = n_splits,
               random_state = random_state,
               shuffle=True)
                   
    for counter,(train_index,test_index) in enumerate(cv.split(X,Y)):
        ### Train/Validation splitter (for Grid Search) ###
        _cv = KFold(n_splits = n_splits_grid_search, 
                    random_state = random_state,
                    shuffle = True)
       
        ### Ridge ###    
        estimator = Ridge(fit_intercept = fit_intercept,
                          solver = 'saga',
                          max_iter = 1000)
        parameters = {'ridge__alpha':lambda_list}
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
        estimator = SmoothLinearRegressor(pyomo_solver,
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
        estimator = RelaxedLassoRegressor(fit_intercept = fit_intercept,
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
        estimator = AdaptiveLassoRegressor(fit_intercept = fit_intercept,
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
        estimator = BARRegressor(fit_intercept = fit_intercept,
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
        estimator = NNGarroteRegressor(pyomo_solver,
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
        estimator = SACRRegressor(solver = pyomo_solver,
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
        

