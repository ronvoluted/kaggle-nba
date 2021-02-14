# -*- coding: utf-8 -*-
import logging
import pandas as pd

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

    X['binned_GP'] = pd.cut(X['GP'], 5, labels=[1,2,3,4,5])
    # X['binned_FG%'] = pd.cut(X['FG%'], 5, labels=[2,1,0,-1,-2], ordered=False)
    # X['binned_3P%'] = pd.cut(X['3P%'], 5, labels=[2,1,0,-1,-2])
    # X['binned_FT%'] = pd.cut(X['FT%'], 5, labels=[2,1,0,-1,-2])
    # *_small features are created when the stat is in lower 25%
    # X['GP_small'] = pd.cut(X['GP'], 5, labels=[2,1,0,-1,-2])
    X['MIN_small'] = pd.cut(X['MIN'], 5, labels=[2,1,0,-1,-2])
    X['PTS_small'] = pd.cut(X['PTS'], 5, labels=[2,1,0,-1,-2])
    X['FTM_small'] = pd.cut(X['FTM'], 5, labels=[2,1,0,-1,-2])
    X['OREB_small'] = pd.cut(X['OREB'], 5, labels=[2,1,0,-1,-2])
    X['AST_small'] = pd.cut(X['AST'], 5, labels=[2,1,0,-1,-2])
    X['BLK_small'] = pd.cut(X['BLK'], 5, labels=[2,1,0,-1,-2])
    # BadStats combines all *_small features and then average them to form an indication.
    # BadStats = 1 means a player performed poorly across the board
    # BadStats = 0 means a player performed not-poorly across the board
    X['BadStats'] = X.apply(lambda row: (row['MIN_small'] + row['PTS_small'] + row['FTM_small'] + row['OREB_small'] + row['AST_small'] + row['BLK_small']) / 6, axis = 1)
    # X['BadStats'] = X.apply(lambda row: (row['FG%_bad'] + row['3P%_bad'] + row['FT%_bad']) / 3, axis = 1)
    # X['NotActive'] = X.apply(lambda row: 1 if (row['FGA']*row['3PA']*row['FTA'] == 0) else 0, axis = 1)
    # X['NotTeamPlayer'] = X.apply(lambda row: 1 if (row['AST']*row['BLK']*row['DREB'] == 0) else 0, axis = 1)

    # X = X.drop(columns=['FGM','FGA','3P Made','3PA','FTA','DREB','REB','TOV','STL'])
    X = X.drop(columns=['MIN_small','PTS_small','FTM_small','OREB_small','AST_small','BLK_small'])
    # X = X.drop(columns=['FG%','3P%','FT%','GP'])
    # X = X.drop(columns=['FG%_bad','3P%_bad','FT%_bad','GP'])

    return X
