## penlm - Penalized Linear Models

`penlm` is a Python package that implements a few penalty based linear models that aren't (currently) available in `scikit-learn`. All the models are implemented with the familiar `fit`/`predict`/`predict_proba`/`score` interface. 

The supported estimators are:
- Smoothly Adaptively Centered Ridge (SACR)
- Ridge penalty on first/second derivatives (Functional Linear Model)
- Adaptive Lasso
- Relaxed Lasso
- Broken Adaptive Ridge (BAR) 
- Non-negative Garrote (NNG)

All the estimators have `fit_intercept` and `scale` (with `sklearn.preprocessing.StandardScaler`) arguments, and work for the following tasks:

- Linear Regression
- Binary Classification (Logistic Regression)
- Multiclass Classification (One-vs-Rest)

A custom cross-validation object is provided in order to perform grid search hyperparameter tuning (with any splitter from `scikit-learn`), and uses `multiprocessing` for parallelization (default `n_jobs = -1`).

## Installation

The package can be installed from terminal with the command `pip install penlm`. Some of the estimators in `penlm` are obtained by directly wrapping `scikit-learn` classes, while the SACR, FLMs, and NNG are formulated using `Pyomo`, which in turn needs a `solver` to interface with. Depending on the estimator, the optimization problems are quadratic with equality and inequality constraints, and all the code was tested using the solver `Ipopt`. On Linux, you just need to download the `Ipopt` executable binary at this [link](https://ampl.com/products/solvers/open-source/#ipopt), and then add the folder that cointains it to your path.


## Usage

The following snippet 

```sh
todo
```


## Citing

todo
