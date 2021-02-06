# -*- coding: utf-8 -*-
import logging
from sklearn.preprocessing import MinMaxScaler
import joblib

def scale(X_train, X_val, X_test):
    """ Use MinMaxScaler to scale training, valuation and test sets, and store them into ../data/processed/
        PARAMETERS:
            X_train: training set (array)
            X_val: valuation set (array)
            X_test: test set (array)
    """
    logger = logging.getLogger(__name__)
    logger.info('Min-max scaling on training, valuation and test sets')

    sc = MinMaxScaler().fit(X_train)
    X_train = sc.transform(X_train)
    X_val = sc.transform(X_val)
    X_test = sc.transform(X_test)

    joblib.dump(X_train, "../data/processed/X_train")
    joblib.dump(X_val, "../data/processed/X_val")
    joblib.dump(X_test, "../data/processed/X_test")