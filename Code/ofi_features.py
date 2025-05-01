import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

def load_preprocess(file_path):
  # 1. Read in the data
  df = pd.read_csv('drive/MyDrive/first_25000_rows.csv')
  df = df.sort_values('ts_event')

  #re-index the series
  df['ts_event'] = pd.to_datetime(df['ts_event'])
  df['ts_recv'] = pd.to_datetime(df['ts_recv'])
  df = df.set_index("ts_event")

  return df 
"""
Computes OF_a or OF_b for a given depth level m. For all time steps.
"""
def compute_OF(df, m, side, supress=False):
    """
    Compute the order‐flow (OF) series for depth level m and side ('b' <-- bid or 'a' <-- ask)
    using the snapshot columns bid_px_## / bid_sz_## or ask_px_## / ask_sz_##.
    Returns a pandas Series aligned with df.index.
    """
    # build the column suffix (00, 01, … 09)
    suffix = f"{m-1:02d}"



    if side.lower() == 'b':
        price_col = f"bid_px_{suffix}"
        size_col  = f"bid_sz_{suffix}"
    else:
        price_col = f"ask_px_{suffix}"
        size_col  = f"ask_sz_{suffix}"
   
     
    try:
      # shift within each symbol to get the "previous" snapshot values
      prev_price = df.groupby('symbol')[price_col].shift(1)
      prev_size  = df.groupby('symbol')[size_col] .shift(1)
    except KeyError:
      if not supress:
        print(f"Order level {m} is too deep or invalid. Please check again.")
      return pd.Series()

    # for the very first row at each symbol/level, treat it as a new level
    prev_price = prev_price.fillna(df[price_col])
    prev_size  = prev_size .fillna(0)

    # apply the three‐case piecewise rule
    of = np.where(
        df[price_col] >  prev_price,        df[size_col],                  # price up: full new queue
        np.where(
          df[price_col] == prev_price,      df[size_col] - prev_size,      # same price: net change
                                            -df[size_col]                  # price down: full removal
        )
    )
    return pd.Series(of, index=df.index)


"""
Computes OF_a or OF_b for all depth levels. For all time steps.
"""
def compute_OF_all_levels(df, side):
    """
    Compute order‐flow (OF) for all available depth levels on one side ('b' or 'a').
    Returns a DataFrame with columns OF_{side}_1, OF_{side}_2, ..., until no more levels exist.
    """
    of_dict = {}
    m = 0
    while True:
        m += 1
     
        of_series = compute_OF(df, m, side, supress=True)
        if(of_series.empty):
          break
        of_dict[f'OF_{m+1}'] = of_series

    return pd.DataFrame(of_dict, index=df.index)

"""
Computes OFI for all depth levels (OFI = OF_b-OF_a). For all time steps.
"""
def compute_OFI_all_levels(df):
    """
    Compute 1-day order‐flow (OFI) for all available depth levels.
    Returns a DataFrame with columns OFI_{side}_1, OFI_{side}_2, ..., until no more levels exist.
    """
    OF_a = compute_OF_all_levels(df, 'a')     #ask OF
    OF_b = compute_OF_all_levels(df, 'b')     #bid OF
    OFI = OF_b - OF_a

    #adjust the columns name for clarity
    new_names = [f"OFI_{m+1}" for m in range(len(OFI.columns))]
    OFI.columns = new_names
    

    return OFI

"""
Computes The normalized OFI for all depth levels. For all time steps. Note that the normalized_OFI is given by the following equation:

ofi_{i,t}^{m,h} = OFI_{i,t}^{m,h} / Q_{i,t}^{m,h}

Where Q_{i,t}^{m,h} = average of <size>_{t,t+h}> cross levels where <.>_{a,b}  is the time averaging operator between a, b.

"""
def compute_normalized_OFI(df, period = 1):
    """
    Compute h-period normalized order‐flow (ofi) for all available depth levels.
    Returns a DataFrame with columns ofi_{side}_1, ofi_{side}_2, ..., until no more levels exist.
    """
    #get the accumulated OFI
    OFI = compute_OFI_all_levels(df)
    total_OFI = OFI.rolling(period).sum()
    

    #we need to also compute the normalization_factor, this involves computing the average mid size over the period
    total_depth = len(OFI.columns)
    mid_sizes_dict = {}
    for m in range(total_depth):
      #compute the mid_size
      mid_size_m = (df[f'bid_sz_{m:02d}'] + df[f'ask_sz_{m:02d}']) / 2
      #save it to the dict
      mid_sizes_dict[f'mid_size_{m+1}'] = mid_size_m
    mid_sizes = pd.DataFrame(mid_sizes_dict,index=df.index)
    #the average mid_size over the period
    mid_size_avg_period = mid_sizes.rolling(period).mean()

    #the normalization is then the average across the levels
    normalization_factor = mid_size_avg_period.mean(axis=1)

    #compute the normalized OFI
    normalized_OFI = OFI.divide(normalization_factor, axis=0)

    new_names = [f"ofi_{m+1}" for m in range(len(OFI.columns))]
    normalized_OFI.columns = new_names
    
    return normalized_OFI

"""
Computes the integrated OFI. For all time steps. We use the PCA weights of the OFI from a historical sense. 
"""
def compute_integrated_OFI(df, start, end, period=1):
  """

  NOTE: According to the paper the integrated OFI uses the PCA weights of the OFI from a historical sense. 
  One has to define the start and end of the historical period.

  df = The loaded dataframe
  start = The start date of the integrated OFI PCA extraction
  end = The end date of the integrated OFI PCA extraction
  period = The period (h)
  
  """
  normalized_ofi = compute_normalized_OFI(df, period)
  #for the start and end we extract the training set
  train_set = normalized_ofi.iloc[start : end]
  #get the PCA
  #fit PCA with a single component
  pca = PCA(n_components=1)
  pca.fit(train_set.values)

  #extract the first‐component weights
  raw_weights = pd.Series(
      pca.components_[0],
      index=normalized_ofi.columns,
      name='PC1_weights'
  )
  #normalize under l1-norm
  l1_norm = raw_weights.abs().sum()
  weights = raw_weights.divide(l1_norm)

  #project the full normalized OFI onto the first component
  integrated = normalized_ofi.dot(weights)

  return (integrated, weights)



"""
Computes the best_level, multi_level and integrated OFI for a given timestamp. Note that cross-asset is not avaliable since the data-set only contains one asset.
"""
def features_at_stamp(df,n,start, end, period=1):
  #compute the integrated and ofi over the whole data-set
  integrated, weights = compute_integrated_OFI(df, start, end, period=1)
  ofi = compute_normalized_OFI(df, period)


  ofi_data_at_stamp = ofi.iloc[n]
  ofi_data_at_stamp["ofi_I"] = integrated.iloc[n]
  ts_event = df.index[n]
  ofi_data_at_stamp = ofi_data_at_stamp.rename(f"Features Data At Stamp {ts_event}")

  return ofi_data_at_stamp




def compute_normalized_OFI_between_timestamps(df, start, end):
    """
    Compute h-period normalized order‐flow (ofi) for all available depth levels.
    Returns a DataFrame with columns ofi_{side}_1, ofi_{side}_2, ..., until no more levels exist.
    """
    #get the accumulated OFI
    OFI = compute_OFI_all_levels(df)
    selected_data = OFI[start : end]
    total_OFI = selected_data.sum()
    

    #we need to also compute the normalization_factor, this involves computing the average mid size over the period
    total_depth = len(OFI.columns)
    mid_sizes_dict = {}
    for m in range(total_depth):
      #compute the mid_size
      mid_size_m = (df[f'bid_sz_{m:02d}'] + df[f'ask_sz_{m:02d}']) / 2
      #save it to the dict
      mid_sizes_dict[f'mid_size_{m+1}'] = mid_size_m
    mid_sizes = pd.DataFrame(mid_sizes_dict,index=df.index)
    #the average mid_size over the period
    mid_size_avg_period = mid_sizes[start: end].mean()
    #the normalization is then the average across the levels
    normalization_factor = mid_size_avg_period.mean()

    #compute the normalized OFI
    normalized_OFI = selected_data.divide(normalization_factor, axis=0)
    

    new_names = [f"ofi_{m+1}" for m in range(len(OFI.columns))]
    normalized_OFI.columns = new_names
    
    return normalized_OFI.sum()

"""
Computes the integrated ofi over two timestamps, start and end. We use ts_event as the measure of the time

training_start : the start time of the training set for PCA extraction
training_end : the end time of the training set for PCA extraction
"""

def compute_integrated_OFI_between_timestamps(df, start, end, training_start, training_end):
  """

  NOTE: According to the paper the integrated OFI uses the PCA weights of the OFI from a historical sense. 
  One has to define the start and end of the historical period.

  df = The loaded dataframe
  start = The start date of the integrated OFI PCA extraction
  end = The end date of the integrated OFI PCA extraction
  period = The period (h)
  
  """
  normalized_ofi = compute_normalized_OFI_between_timestamps(df, start, end)
  #we do PCA over one day normalized OFI 
  one_day_normalized_ofi =  compute_normalized_OFI(df)
  train_set = one_day_normalized_ofi[training_start : training_end]
  #get the PCA
  #fit PCA with a single component
  pca = PCA(n_components=1)
  pca.fit(train_set.values)

  #extract the first‐component weights
  raw_weights = pd.Series(
      pca.components_[0],
      index=train_set.columns,
      name='PC1_weights'
  )
  #normalize under l1-norm
  l1_norm = raw_weights.abs().sum()
  weights = raw_weights.divide(l1_norm)

  #project the full normalized OFI onto the first component
  integrated = normalized_ofi.dot(weights)

  return (integrated, weights)




  
"""
Computes the non-normalized ofi over two timestamps, start and end. We use ts_event as the measure of the time
"""
def compute_OFI_between_timestamps(df, start, end):
  #compute the normalized ofi with period one
  ofi = compute_OFI_all_levels(df)
  #we then get the ofi data between start and end
  ofi_data_between_timestamps = ofi[start : end]
  #we get the total orderflow between those time-stamp and hence we sum them
  ofi_data_between_timestamps = ofi_data_between_timestamps.sum()

  return ofi_data_between_timestamps


"""
Cross asset feature. Suppose that we have a list of dataframes, one for each asset. For each one we compute the ofi features between two time stamps.

df_list : list of the dataframes for each asset
start : the start time of opi period (same for all)
end : the end time of the opi period (same for all)
training_start : the list of start time of the training set for PCA extraction
training_end : the list of end time of the training set for PCA extraction
weights : the weights of the cross asset ofi
"""
def compute_cross_asset_features(df_list, start, end, training_start, training_end, weights):
  ofi_dataset = {}
  COFI = 0
  for i in range(len(df_list)):
    #get the symbol
    df = df_list[i]
    asset_name = df["symbol"].iloc[0]
    #get the ofi data
    ofi_data = compute_normalized_OFI_between_timestamps(df, start, end)
    #add the integrated ofi
    integrated_ofi, pca_weights = compute_integrated_OFI_between_timestamps(df, start, end, training_start[i], training_end[i])
    ofi_data["ofi_I"] = integrated_ofi

    COFI = weights[i] * ofi_data

    ofi_dataset[asset_name] = ofi_data

  #we then add a column called the COFI the cross asset OFI
  ofi_dataset["COFI"] = COFI
 

  frame = pd.DataFrame(ofi_dataset)

  return frame 
