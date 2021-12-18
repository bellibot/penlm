## penlm - Penalized Linear Models

`penlm` is a Python package that implements a few penalty based linear models that aren't (currently) available in `scikit-learn`. All the models are implemented with the familiar `fit`/`predict`/`predict_proba`/`score` interface. 

The supported estimators are:
- Smoothly Adaptively Centered Ridge ([SACR](https://doi.org/10.1016/j.jmva.2021.104882))
- Ridge penalty on first/second derivatives (Functional Linear Model)
- [Adaptive Lasso](https://doi.org/10.1198/016214506000000735)
- [Relaxed Lasso](https://doi.org/10.1016/j.csda.2006.12.019)
- Broken Adaptive Ridge ([BAR](https://doi.org/10.1016/j.jmva.2018.08.007)) 
- Non-negative Garrote ([NNG](https://doi.org/10.2307/1269730))

All the estimators have `fit_intercept` and `scale` (with `sklearn.preprocessing.StandardScaler`) arguments, and work for the following tasks:

- Linear Regression
- Binary Classification (Logistic Regression)
- Multiclass Classification (One-vs-Rest)

A custom cross-validation object is provided in order to perform grid search hyperparameter tuning (with any splitter from `scikit-learn`), and uses `multiprocessing` for parallelization (default `n_jobs = -1`).

## Installation

The package can be installed from terminal with the command `pip install penlm`. Some of the estimators in `penlm` are obtained by directly wrapping `scikit-learn` classes, while the SACR, FLMs, and NNG are formulated using `Pyomo`, which in turn needs a `solver` to interface with. Depending on the estimator, the optimization problems are quadratic with equality and/or inequality constraints, and all the code was tested using the solver [Ipopt](https://doi.org/10.1007/s10107-004-0559-y). You just need to download the [executable binary](https://ampl.com/products/solvers/open-source/#ipopt), and then add the folder that contains it to your path (tested on Ubuntu).


## Usage

The following snippet shows how to fit an estimator with its own parameters and grid search object:

```sh
import numpy as np
import penlm.grid_search as gs
from penlm.smoothly_adaptively_centered_ridge import SACRClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_wine

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
                              scoring = 'balanced_accuracy')
grid_search.fit(X[train_index],
                Y[train_index])
score = grid_search.best_estimator.score(X[test_index],
                                         Y[test_index],
                                         scoring = 'balanced_accuracy',
                                         verbose = False,
                                         n_jobs = -1)
```
A test script for all estimators (in both classification and regression) is also provided in the github repo. 

## Citing

We encourage the users to cite the original papers of the implemented estimators. 
In particular, the code published in this package has been used in the case studies of [this](https://doi.org/10.1016/j.jmva.2021.104882) paper.
