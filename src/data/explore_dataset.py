# -*- coding: utf-8 -*-
import logging
import pandas as pd

def explore(filename):
    """ Load the raw data and print out information and dataframe
    """

    logger = logging.getLogger(__name__)
    logger.info('explore data from path: data/raw/'+filename)

    pd.set_option('display.max_columns', None)
    df = pd.read_csv("../data/raw/"+filename)
    
    print("=== dataframe info ===")
    print(df.info(), end="\n")
    print("=== dataframe shape ===")
    print(df.shape, end="\n")
    if 'TARGET_5Yrs' in df:
        print("=== Target Value Count ===")
        print(df['TARGET_5Yrs'].value_counts(), end="\n")
    print("=== dataframe describe ===")
    print(df.describe(include='all'), end="\n")
    




if __name__ == '__explore__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    explore()
