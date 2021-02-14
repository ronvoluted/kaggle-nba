import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib

def blend(X, y, X_check, y_check, X_test, clfs, filename):
    """Blending a collection of classifiers:
    This module is to blend a list of already trained classifiers as stage 1, 
    and use their output(predict_proba) as input weights for stage 2 RandomForestClassifer.
    By doing this, it can utilize the insights from the collection to form a stronger classifier.

    Parameters
    ----------
    X : dataframe or array
        Features of training dataset
    y : dataframe or array
        Target value of training dataset
    X_check : dataframe or array
        Features of validation dataset
    y_check : dataframe or array
        Target value of validation dataset
    X_test : dataframe or array
        Features of test set
    clfs : list
        A collection of trained classifiers.
    filename : string
        The filename of the final submission csv
    
    Returns
    -------
    """
    np.random.seed(0)  # seed to shuffle the train set
    n_folds = 10
    verbose = True
    shuffle = False

    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]

    skf = StratifiedKFold(n_folds)
    print(skf)

    print("Creating train, validation and test sets for blending.")

    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_check = np.zeros((X_check.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_test.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):
        # print(j, clf)
        dataset_blend_test_j = np.zeros((X_test.shape[0], n_folds))
        dataset_blend_check_j = np.zeros((X_check.shape[0], n_folds))
        for i, (train, val) in enumerate(skf.split(X,y)):
            # print("Fold", i)
            X_train = X[train]
            y_train = y[train]
            X_val = X[val]
            y_val = y[val]
            y_submission = clf.predict_proba(X_val)[:, 1]
            dataset_blend_train[val, j] = y_submission
            dataset_blend_check_j[:, i] = clf.predict_proba(X_check)[:, 1]
            dataset_blend_test_j[:, i] = clf.predict_proba(X_test)[:, 1]
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
        dataset_blend_check[:, j] = dataset_blend_check_j.mean(1)

    print("Blending.")
    clf = RandomForestClassifier(max_depth=3)
    clf.fit(dataset_blend_train, y)
    joblib.dump(clf,"../models/kpw_best_blending_model_assignmentB")

    print("==== ROC AUC Score for training set ====")
    print(roc_auc_score(y,clf.predict_proba(dataset_blend_train)[:, 1]))

    print("==== ROC AUC Score for valuation set ====")
    y_check_sub = clf.predict_proba(dataset_blend_check)[:, 1]
    y_check_sub = (y_check_sub - y_check_sub.min()) / (y_check_sub.max() - y_check_sub.min())
    print(roc_auc_score(y_check,y_check_sub))


    y_submission = clf.predict_proba(dataset_blend_test)[:, 1]

    print("Linear stretch of predictions to [0,1]")
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

    print("Saving Results.")
    tmp = np.vstack([range(0, len(y_submission)), y_submission]).T
    np.savetxt(fname=filename, X=tmp, fmt='%d,%0.9f',
               header='Id,TARGET_5Yrs', comments='')