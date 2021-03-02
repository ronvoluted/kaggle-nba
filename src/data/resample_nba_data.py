from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import pandas as pd
def resample_nba_data(df):
    """
    
    This module is specific to the  Kaggle NBA Competition set up for the Adv Data Science class.
    
    Upscaling and Downscaling performed on the dataframe.
    
    Takes dataframe as input, based on the majority and minority Target Class, 
    the function will upscale the data and downscale the data and return two dataframes. 
    
    It returns two dataframes.
    
    Parameters
    ----------
    df : dataframe
    
    """
    
    df_majority = df[df['TARGET_5Yrs']==1]
    df_minority = df[df['TARGET_5Yrs']==0]
    
    print('df_minority.shape',df_minority.shape,' df_majority.shape' , df_majority.shape)
    print('df_minority.shape',df_minority['TARGET_5Yrs'].value_counts(),' df_majority.shape' , df_majority['TARGET_5Yrs'].value_counts())
    
    df_minority_upsampled = resample(df_minority,replace=True,n_samples=df_majority.shape[0],random_state=123) 
    df_majority_downsampled = resample(df_majority,replace=True,n_samples=df_minority.shape[0],random_state=123) 
    
    df_upsampled = pd.concat([df_majority, df_minority_upsampled],ignore_index=True)
    print('df_upsampled ',df_upsampled['TARGET_5Yrs'].value_counts())
    print(df_upsampled['TARGET_5Yrs'].value_counts())
    
    df_downsampled = pd.concat([df_majority_downsampled, df_minority],ignore_index=True)
    print('df_downsampled ',df_downsampled['TARGET_5Yrs'].value_counts())
    print(df_downsampled['TARGET_5Yrs'].value_counts())

    return df_upsampled, df_downsampled