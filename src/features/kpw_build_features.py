# -*- coding: utf-8 -*-
import logging

def build(X):
    """Build Features:
    This module is to create new features from existing features

    Parameters
    ----------
    X : dataframe
        Features of dataset
    
    Returns
    -------
    X
        New dataframe with new features
    """
    logger = logging.getLogger(__name__)
    logger.info('Build features')

    # df = df.drop(columns=['FG%','3P%','FT%'])

    X['MPG'] = X.apply(lambda row: row['MIN'] / row['GP'], axis = 1)
    # X['FG%'] = X.apply(lambda row: row['FGM'] / row['FGA'] if row['FGA'] > 0 else 0, axis = 1)
    X['FG%_Rank'] = X.apply(lambda row: 1 if row['FG%'] < 0.5 else 0, axis = 1)
    # X['3P%'] = X.apply(lambda row: row['3P Made'] / row['3PA'] if row['3PA'] > 0 else 0, axis = 1)
    X['3P%_Rank'] = X.apply(lambda row: 1 if row['3P%'] < 0.5 else 0, axis = 1)
    # X['FT%'] = X.apply(lambda row: row['FTM'] / row['FTA'] if row['FTA'] > 0 else 0, axis = 1)
    X['FT%_Rank'] = X.apply(lambda row: 1 if row['FT%'] < 0.5 else 0, axis = 1)
    # X['GP_small'] = X.apply(lambda row: 1 if row['GP'] < 50 else 0, axis = 1)
    # X['GP_medium'] = X.apply(lambda row: 1 if (row['GP'] >= 50 and row['GP'] < 100) else 0, axis = 1)
    # X['MIN_small'] = X.apply(lambda row: 1 if row['MIN'] <= 20 else 0, axis = 1)
    # X['MIN_medium'] = X.apply(lambda row: 1 if (row['MIN'] > 20  and row['MIN'] <= 40) else 0, axis = 1)
    # X['NotActive'] = X.apply(lambda row: 1 if (row['FGA']+row['3PA']+row['FTA']+row['OREB']+row['STL'] <= 8) else 0, axis = 1)
    X['NotActive'] = X.apply(lambda row: 1 if (row['FGA']*row['3PA']*row['FTA']*row['OREB']*row['STL'] == 0) else 0, axis = 1)
    # X['NotTeamPlayer'] = X.apply(lambda row: 1 if (row['AST']+row['BLK']+row['DREB'] <= 3) else 0, axis = 1)
    X['NotTeamPlayer'] = X.apply(lambda row: 1 if (row['AST']*row['BLK']*row['DREB'] == 0) else 0, axis = 1)
    # X['REB'] = X.apply(lambda row: row['OREB'] + row['DREB'], axis = 1)
    # X['GP_large'] = X.apply(lambda row: 1 if (row['GP'] > 100) else 0, axis = 1)
    # X = X.drop(columns=['MIN','GP','FG%','3P%','FT%'])
    # X = X.drop(columns=['FG%','3P%','FT%'])

    return X
