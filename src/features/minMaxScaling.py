# -*- coding: utf-8 -*-
import logging
from sklearn.preprocessing import MinMaxScaler
import joblib

def scale(X_train, X_val):
    """ Use MinMaxScaler to scale training, valuation and test sets, and store them into ../data/processed/
        PARAMETERS:
            X_train: training set (array)
            X_val: valuation set (array)
    """
    logger = logging.getLogger(__name__)
    logger.info('Min-max scaling on training, valuation and test sets')

    sc = MinMaxScaler().fit(X_train)
    X_train = sc.transform(X_train)
    X_val = sc.transform(X_val)

    return sc, X_train, X_val
