"""libs/data_utils.py
Author: Adam J. Vogt (Aug. 2017)
----------

utils for extracting, saving, and creating data set

"""

import pandas as pd


def get_data():
    """Load Congressional Voting Records Data
    Opens csv if available in working directory,
    otherwise downloads data set and 
    saves to csv in working directory
    Parameters
    ----------
    none
    
    Returns
    -------
    df : pd.DataFrame
        Pandas Data Frame for voting record
    
    """
    
    try:
        df = pd.read_csv('house_votes_84.csv')
    except:
        print("CSV not found, downloading data...")
        df = pd.read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data',
            header=None)
        df.to_csv('house_votes_84.csv', index=False)
    
    return df


def df_to_array(df):
    """Convert DataFram to Array
    One hot encodes voting data and exports to numpy arrays
    Parameters
    ----------
    df : pd.DataFrame
        the original data frame for the voting records
    
    Returns
    -------
    X : np.ndarray, shape = [438, 48]
        One hot encoded array of votes cast for each measure
    y : np.ndarray, shape = [438]
        One hot encoded array for party affiliation
        (democrat = 0, republican = 1)
    
    """
    X = pd.get_dummies(df.iloc[:, 1:]).values
    
    from sklearn.preprocessing import LabelEncoder
    party_le = LabelEncoder()
    y = df.iloc[:, 0].values
    y = party_le.fit_transform(y)
    
    return X, y
