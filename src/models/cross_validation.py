import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.features import minMaxScaling
from src.models import eval_model

def cv(classifier, X, y):
    n = 5
    skf = StratifiedKFold(n_splits = n)
    roc_training = np.array([])
    roc_val = np.array([])

    for train_index, val_index in skf.split(X, y):
        X_train = X[train_index]
        X_val = X[val_index]
        sc, X_train, X_val = minMaxScaling.scale(X_train, X_val)
        y_train, y_val = y[train_index], y[val_index]

        model, roc_score_training, roc_score_val = eval_model.eval_model(classifier, X_train, y_train, X_val, y_val, show=False)
        roc_training = np.append(roc_training,roc_score_training)
        roc_val = np.append(roc_val, roc_score_val)
    
    roc_training_avg = np.average(roc_training)
    roc_val_avg = np.average(roc_val)

    print("Avg ROC AUC score of training set is: "+str(roc_training_avg))
    print("Avg ROC AUC score of valuation set is: "+str(roc_val_avg))

    return roc_training_avg, roc_val_avg
