from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
from matplotlib.pyplot import xticks
import numpy as np
def plot_validation_curve(estimator,hyperparameter,hyperparameter_value,x,y,title,cv=5,scoring='roc_auc'):
    """
    Runs Validation Curve for Train and Validation set.
    Plots the train and test scores
    
    Parameters
    ----------
       
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.
        
    hyperparameter : hyperparameter name 
        Name of the hyperparameter for plotting validation curve
    
    hyperparameter_value : list of values
        List of values for the hyperparameter selected for tuning
        
    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.
        
    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.
        
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
        
    title : str
        Title for the chart.

    
    """
    
    lw = 2
    
    train_scores, valid_scores = validation_curve(estimator=estimator,scoring=scoring,param_name=hyperparameter,param_range=hyperparameter_value,X=x,y=y,cv=cv)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(valid_scores, axis=1)
    test_scores_std = np.std(valid_scores, axis=1)

    plt.title("Validation Curve with "+title)
    plt.xlabel(hyperparameter)
    plt.ylabel("Score")

    plt.semilogx(hyperparameter_value, train_scores_mean, label="Training score",color="darkorange", lw=lw)
    plt.fill_between(hyperparameter_value, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="darkorange", lw=lw)
    plt.semilogx(hyperparameter_value, test_scores_mean, label="Cross-validation score",color="navy", lw=lw)
    plt.fill_between(hyperparameter_value, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="navy", lw=lw)
    plt.xticks(ticks=hyperparameter_value,labels=hyperparameter_value,rotation = 'vertical')
    plt.legend(loc="best")
    plt.show()
    
    return