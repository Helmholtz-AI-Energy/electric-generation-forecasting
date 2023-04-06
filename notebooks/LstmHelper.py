import pandas as pd
import numpy as np

def scale_data(df, individual=True):
    """
    Scale input data.
    Params
    ------
    df : pandas.DataFrame
         Data to scale.
    individual : bool
                 Whether to scale each feature individually or all features globally.
    Returns
    -------
    pandas.DataFrame : Scaled data.
    """
    
    if individual is True:
        print("Scale each feature individually.")
        scaler = MinMaxScaler().fit(df.to_numpy(dtype=float))
        norm_df = pd.DataFrame(scaler.transform(df.to_numpy(dtype=float)))
        norm_df.columns = df.columns
            
    else:
        print("Scale features globally.")
        norm_df = df.copy() # Copy original data frame for storing scaled data.
        scaler = MinMaxScaler().fit(df.to_numpy(dtype=float).flatten().reshape(-1, 1))
        for x in df.columns:
            norm_df[x] = scaler.transform(df[x].to_numpy(dtype=float).reshape(-1, 1))
    return norm_df, scaler


def unscale_data(df, scaler, individual=True):
    """
    Un-scale input data.
    Params
    ------
    df : pandas.DataFrame
         Data to un-scale.
    individual : bool
                 Un-scale each feature individually or not.
    Returns
    -------
    pandas.DataFrame : Scaled data.
    """
    
    if individual is True:
        print("Un-scale each feature individually.")
        unscaled_df = pd.DataFrame(scaler.inverse_transform(df.to_numpy(dtype=float)))
        unscaled_df.columns = df.columns
            
    else:
        print("Un-scale features globally.")
        unscaled_df = df.copy() # Copy original data frame for storing scaled data.
        for x in df.columns:
            unscaled_df[x] = scaler.inverse_transform(df[x].to_numpy(dtype=float).reshape(-1, 1))
    return unscaled_df


def generate_sequences(df: pd.DataFrame, tw: int, pw: int, target_columns, drop_targets=False):
    """
    Create sequences from univariate time series.
    Params
    ------
    df: pd.DataFrame
        time series data
    tw: int
        training window defining how many steps to look back
    pw: int
        prediction Window defining how many steps forward to predict
  
    Returns
    -------
    Dictionary of sequences and targets.
    """
    data = dict() # Store results in nested dict.
    for i in range(len(df) - tw):
        # Get current sequence  
        sequence = df[i:i+tw].to_numpy(dtype=float)
        # Get values right after the current sequence
        target = df[i+tw:i+tw+pw].to_numpy(dtype=float)
        data[i] = {'sequence': sequence, 'target': target}
    return data