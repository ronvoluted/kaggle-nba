import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from src.features.recalculate_percentage import replaceAll
from src.data.resampling import smote, random_under, oversample

def engineer(df, target, features, predicting=False):
    """engineer:
    Perform data preparation and feature engineering
    
    Parameters
    ----------
    df : dataframe, array
        X data
    target : dataframe, array
        y data
    features : dict
        Dictionary of which data preparation/feature engineering actions to perform
    predicting: bool
        Set True if using a test set. Used to prevent fitting/sampling that is only appropriate for the training set

    Returns
    -------
    df : dataframe
        Dataframe with actions performed
    target : dataframe or array
        Target with actions performed
    """

    # Impute negative values
    if features['new_neg']:
        df_cols = df.columns
        df[df < 0] = np.nan
        imp_mean = IterativeImputer(min_value=0)
        df = pd.DataFrame(imp_mean.fit_transform(df), columns=df_cols)

    # Balance dataset
    if predicting == False:
        if features['new_upsample']:
            df, target = smote(df, target)

        if features['new_downsample']:
            df, target = random_under(df, target)
    
    # Replace percentages with recalculations
    if features['new_pct']:
        df = replaceAll(df)
    
    # Approximation of player possessions using limited dataset stats
    possessions = df['REB'] + df['STL']

    fieldGoalsPer100Min = df['FGA'] / df['MIN'] * 100
    threePointsersPer100Min = df['3PA'] / df['MIN'] * 100
    freeThrowsPerGame = df['FTA'] / df['GP']
    assistsPer100Min = df['AST'] / df['MIN'] * 100
    turnoversPer100Min = df['TOV'] / df['MIN'] * 100
    
    fieldGoalsPerPossession = df['FGA'] / possessions
    threePointsersPerPossession = df['3PA'] / possessions
    freeThrowsPerGame = df['FTA'] / df['GP'] * 100
    assistsPerPossession = df['AST'] / possessions
    turnoversPerPossession = df['TOV'] / possessions

    # Add Possessions column
    if features['add_POS']:
        df['POS'] = possessions

    # Add Points Per Possession column
    if features['add_PPP']:
        df['PPP'] = df['GP'] * df['PTS'] / possessions

    # Add 3-Pointers Per Possession column
    if features['add_3PP']:
        df['FGP'] = df['3P Made'] / (100 * possessions)

    # Add Field Goals Per Possession column
    if features['add_FGP']:
        df['FGP'] = df['FGM'] / (100 * possessions)

    # Add Free Throws per Game column
    if features['add_FTG']:
        df['FGP'] = df['FTM'] / df['GP']

    # Add 3-Pointers Reliable column
    if features['add_3PR']:
        df['3PR'] = df['3P%'] > df['3P%'].mean() * 0.75
    
    # Add Field Goals Reliable column
    if features['add_FGR']:
        df['FGR'] = df['FG%'] > df['FG%'].mean() * 0.75
    
    # Add Free Throws Reliable column
    if features['add_FTR']:
        df['FTR'] = df['FT%'] > df['FT%'].mean() * 0.75

    # Relative Offense Number (R.O.N.)
    # Approximation of Usage Rating from NBA analysts, using the more limited stats available in the dataset        

    # Add RON per 100 Minutes
    if features['add_RONM']:
        df['RONM'] = (fieldGoalsPer100Min + threePointsersPer100Min) * freeThrowsPerGame * assistsPer100Min * turnoversPer100Min
    
    # Add RON per Possessions
    if features['add_RONP']:
        df['RONP'] = (fieldGoalsPerPossession + threePointsersPerPossession) * df['FTA'] * assistsPerPossession * turnoversPerPossession

    # Remove Points per Game column
    if features['rem_PTS']:
        df = df.drop('PTS', 1)

    # Remove goal percentage columns
    if features['rem_pct']:
        df = df.drop(['3P%', 'FG%', 'FT%'], 1)

    return df, target
