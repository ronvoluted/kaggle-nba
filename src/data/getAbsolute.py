# -*- coding: utf-8 -*-
import logging
import pandas as pd
import joblib

def abs(df, name):
    """ Convert all value in dataframe to be absolute value, 
        and store the dataframe into ../data/processed/abs_<name>
    """
    logger = logging.getLogger(__name__)
    logger.info('turning dataframe '+name+ ' to be absolute value')

    df = df.abs()
    joblib.dump(df, "../data/processed/abs_"+name)
    return df
