## penlm - Penalized Linear Models

`penlm` is a Python package that implements a few penalty based linear models that aren't (currently) available in `scikit-learn`. All the models are implemented with the familiar `fit`/`predict`/`predict_proba`/`score` interface.

The supported estimators are:
- Smoothly Adaptively Centered Ridge ([SACR](https://doi.org/10.1016/j.jmva.2021.104882))
- Ridge penalty on first/second derivatives (Functional Linear Model)
- [Adaptive Lasso](https://doi.org/10.1198/016214506000000735)
- [Relaxed Lasso](https://doi.org/10.1016/j.csda.2006.12.019)
- Broken Adaptive Ridge ([BAR](https://doi.org/10.1016/j.jmva.2018.08.007)) 
- Non-negative Garrote ([NNG](https://doi.org/10.2307/1269730))

All the estimators have `fit_intercept` and `scale` arguments (scaling is done with `sklearn.preprocessing.StandardScaler`) , and work for the following tasks:

- Linear Regression
- Binary Classification (Logistic Regression)
- Multiclass Classification (One-vs-Rest)

A custom `cross-validation` object is provided in order to perform grid search hyperparameter tuning (with any splitter from `scikit-learn`), and uses `multiprocessing` for parallelization (default `n_jobs = -1`).
Multiclass fitting is `not` parallelized in this version (that would be beneficial when a high number of cores is available, or when refitting the best estimator in the grid search object).

## Installation

The package can be installed from terminal with the command `pip install penlm`. Some of the estimators in `penlm` are obtained by directly wrapping `scikit-learn` classes, while the SACR, FLMs, and NNG are formulated using `Pyomo`, which in turn needs a `solver` to interface with. Depending on the estimator, the optimization problems are quadratic with equality and/or inequality constraints, and all the code was tested using the solver [Ipopt](https://doi.org/10.1007/s10107-004-0559-y). You just need to download the [executable binary](https://ampl.com/products/solvers/open-source/#ipopt), and then add the folder that contains it to your path (tested on Ubuntu).


## Usage

The following lines show how to fit an estimator with its own parameters and grid search object, by using a `StratifiedKFold` splitter:

```sh
import numpy as np
import penlm.grid_search as gs
from penlm.smoothly_adaptively_centered_ridge import SACRClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_wine
from sklearn.metrics import balanced_accuracy_score

X, Y = load_wine(return_X_y = True)

train_index = [i for i in range(100)]
test_index = [i for i in range(100, len(X))]

lambda_list = np.logspace(-5, 5, 10, base = 2)
phi_list = np.linspace(0, 1, 10)[1:]


estimator = SACRClassifier(solver = 'ipopt',
                           scale = True,
                           fit_intercept = True)
parameters = {'phi':phi_list,
              'lambda':lambda_list}
cv = StratifiedKFold(n_splits = 2, 
                     random_state = 46,
                     shuffle = True)              
grid_search = gs.GridSearchCV(estimator,
                              parameters,
                              cv,
                              scoring = balanced_accuracy_score)
grid_search.fit(X[train_index],
                Y[train_index])
score = grid_search.score(X[test_index],
                          Y[test_index],
                          scoring = balanced_accuracy_score,
                          verbose = False,
                          n_jobs = -1)
```
The test folder in the `github` repo contains two sample scripts that show how to use all the estimators (in both classification and regression tasks). In particular, for the Adaptive Lasso and the NNG you need to provide an initial coefficient vector as a `np.ndarray`, with the same shape as the one found in `scikit-learn` estimators (the test scripts fit a LogisticRegression/Ridge estimator and use the `estimator.coef_` object).
Moreover, regarding the `scoring`, all the estimators and the grid search class use `accuracy`/`R^2` as default scores (when the argument `scoring = None`), but you can provide any `Callable` scoring function found in `sklearn.metrics`. Beware that higher is better, and therefore when scoring with errors like `sklearn.metrics.mean_squared_error`, you need to wrap that in a custom function that changes its sign.

## Citing

We encourage the users to cite the original papers of the implemented estimators. 
In particular, the code published in this package has been used in the case studies of [this](https://doi.org/10.1016/j.jmva.2021.104882) paper.
