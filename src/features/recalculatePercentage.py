# -*- coding: utf-8 -*-
import logging
import pandas as pd

def replaceAll(df):
    """ replace all percentage with newly calculated value FG%, 3P% and FT%
        PARAMETERS:
            df: dataframe of the dataset
    """
    logger = logging.getLogger(__name__)
    logger.info('replace all percentage with newly calculated value FG%, 3P% and FT%')

    df = df.drop(columns=['FG%','3P%','FT%'])

    df['FG%'] = df.apply(lambda row: row['FGM'] / row['FGA'] if row['FGA'] > 0 else 0, axis = 1)
    df['3P%'] = df.apply(lambda row: row['3P Made'] / row['3PA'] if row['3PA'] > 0 else 0, axis = 1)
    df['FT%'] = df.apply(lambda row: row['FTM'] / row['FTA'] if row['FTA'] > 0 else 0, axis = 1)

    return df