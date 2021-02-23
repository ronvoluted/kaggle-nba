from sklearn.model_selection import train_test_split
from data import split_data as split
def clean_and_split_nba_data(df,inverse=False):
    """
    Imports the file and splits it into Train, Valdiation and Test set.
    It returns all splits.
    
    Parameters
    ----------
    file : input csv file with the path
    """
    
    df_cleaned = df.copy()
    df_cleaned[ df_cleaned<0 ] = 0
    df_cleaned.loc[df_cleaned['3P Made'] > df_cleaned['3PA'], ['3P Made' , '3PA', 'CALC3P%']] = 0, 0, 0
    df_cleaned.loc[df_cleaned['FGM'] > df_cleaned['FGA'], ['FGM', 'FGA', 'CALCFG%']] = 0, 0, 0
    df_cleaned.loc[df_cleaned['FTM'] > df_cleaned['FTA'], ['FTM', 'FTA', 'CALCFT%']] = 0, 0, 0
    df_cleaned.loc[df_cleaned['3P Made'] > 0, ['CALC3P%']] = df_cleaned['3P Made']/df_cleaned['3PA']*100
    df_cleaned.loc[df_cleaned['FGM'] > 0, ['CALCFG%']] =df_cleaned['FGM']/df_cleaned['FGA']*100
    df_cleaned.loc[df_cleaned['FTM'] > 0, ['CALCFT%']] = df_cleaned['FTM']/df_cleaned['FTA']*100
    df_cleaned = df_cleaned.drop(['3P%','FT%','FG%','Id_old','Id'],axis=1)
    df_cleaned = df_cleaned.fillna(0)
    
    if inverse==True:
        df_cleaned['TARGET_5Yrs'] = df_cleaned['TARGET_5Yrs'].replace([0,1],[1,0])
        
    for cols in df_cleaned.columns:
        chk_rows = df_cleaned[df_cleaned[cols]<0].shape[0]
        if chk_rows > 0 :
            print(f'Column Name {cols},\tRows with Negative Value {chk_rows},\tPercentage {chk_rows/len(df)*100}')

    #x_data, x_train, x_val, x_test, y_data , y_train, y_val,  y_test = split.split_data(df,'TARGET_5Yrs')
    x=df_cleaned.drop(['TARGET_5Yrs'],axis=1)
    y=df_cleaned['TARGET_5Yrs']
    x_data , x_test ,y_data,  y_test = train_test_split(x, y, test_size=0.2, random_state = 8, stratify=y)
    x_train , x_val , y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state = 8, stratify=y_data)
    print(x_data.columns)
    return x_data, x_train, x_val, x_test, y_data , y_train, y_val,  y_test