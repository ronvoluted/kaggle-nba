# -*- coding: utf-8 -*-
import logging
import pandas as pd
import numpy as np
from sklearn.utils import resample

def resample(df):
    """ Oversample the dataset to handle imbalanced data
        PARAMETERS:
            df: dataframe of the dataset
    """
    logger = logging.getLogger(__name__)
    logger.info('Oversample the dataset to handle imbalanced data')

    df_1 = df.loc[df['TARGET_5Yrs']==1]
    df_1_len = len(df_1.index)
    df_0 = df.loc[df['TARGET_5Yrs']==0]
    df_0_len = len(df_0.index)

    if ( df_1_len > df_0_len ):
        df_0 = resample(df_0, replace=True, n_samples=df_1_len, random_state=123)
    else:
        df_1 = resample(df_1, replace=True, n_samples=df_0_len, random_state=123)

    df = df_1.append(df_0, ignore_index=True)
    return df