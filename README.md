
# OFI Feature Generation Documentation
Example Usage:
```python
# load into dataframe
df = load_preprocess("first_25000_rows.csv")
# compute the single asset features for a specific index
target_index = 100
# set the PCA extraction start and end date
training_start, training_end = 0, 1000
features = features_at_stamp(df, target_index, training_start, training_end)

# print baselevel ofi and integrated ofi
print(features["ofi_1"], features["ofi_I"])

# suppose that we have more than one asset data, we can compute cross asset features. 
df_2 = load_preprocess("MSFT_data.csv")
# due to potential timestamp discrepancy we avoid using iloc indexing,
# choose a start and end time stamp.
start_time = "2024-10-21 11:54:29.764673165+00:00"
end_time   = "2024-10-21 13:04:16.583527688+00:00"

# we choose a training index for PCA extraction
training_start = [0,0]
training_end   = [1000,1000]

# suppose that we have the cross asset weights from LASSO regressions
weights = [0.1, 0.9]

# then perform the feature generation for the given time frame
features_cross_assets = compute_cross_asset_features(
    [df, df_2], start_time, end_time, training_start, training_end, weights
)

# get asset 1 best level ofi
print(features_cross_assets["AAPL"]["ofi_1"])
# get asset 2 best level ofi
print(features_cross_assets["MSFT"]["ofi_1"])
# get the cross level ofi for level 1
print(features_cross_assets["COFI"]["ofi_1"])
```  

---

### `compute_OF_all_levels(df, side)`

Compute order-flow (OF) series for all available depth levels on one side, for a single asset.

Parameters
----------
df : pandas.DataFrame
    Order book snapshot data as above.

side : {'b', 'a'}
    Book side to compute (`'b'` or `'a'`).

Returns
-------
pandas.DataFrame
    Columns `OF_1, OF_2, ..., OF_M` for each level until no more levels exist.
    Indexed by the same timestamps as `df`.

Example
-------
```python
of_bid = compute_OF_all_levels(df, side='b')
```  

---

### `compute_OFI_all_levels(df)`

Compute raw Order Flow Imbalance (OFI) at all depth levels. For a single asset.

Parameters
----------
df : pandas.DataFrame
    Order book snapshot data.

Returns
-------
pandas.DataFrame
    Columns `OFI_1, OFI_2, ..., OFI_M`, where each is
    `OF_b_m - OF_a_m` for levels 1…M.

Example
-------
```python
ofi = compute_OFI_all_levels(df)
```  

---

### `compute_normalized_OFI(df, period=1)`

Compute normalized OFI over a rolling window. 

Parameters
----------
df : pandas.DataFrame
    Order book snapshot data.

period : int, default 1
    Rolling window length (in updates) for summing OFI and averaging mid-sizes.

Returns
-------
pandas.DataFrame
    Columns `ofi_1, ofi_2, ..., ofi_M`
    where each level’s OFI is divided by the time-and-level average
    of mid-queue sizes over the window.

Example
-------
```python
norm_ofi = compute_normalized_OFI(df, period=5)
```  

---

### `compute_integrated_OFI(df, start, end, period=1)`

Compute the Integrated OFI by PCA projection.

Parameters
----------
df : pandas.DataFrame
    Order book snapshot data.

start, end : int or label
    Historical window (inclusive) for fitting PCA on normalized OFI.

period : int, default 1
    Rolling window length for normalization.

Returns
-------
integrated : pandas.Series
    Single-component OFI time series (projection on first PC), indexed as `df`.

weights : pandas.Series
    L1-normalized PCA loadings for each level (`ofi_1…ofi_M`).

Example
-------
```python
integ, w = compute_integrated_OFI(df, start=0, end=1000, period=10)
```  

---

### `features_at_stamp(df, n, start, end, period=1)`

Extract all feature values at a single update index.

Parameters
----------
df : pandas.DataFrame
    Order book snapshot data.
n : int
    Integer index of the desired timestamp.
start, end : int or label
    PCA training window for integrated OFI.
period : int, default 1
    Rolling window length for normalization.

Returns
-------
pandas.Series
    Feature vector at the nth update:
    `[ofi_1, …, ofi_M, ofi_I]`, renamed with actual timestamp.

Example
-------
```python
feat = features_at_stamp(df, n=2900, start=0, end=1000, period=1)
```  

---

### `compute_normalized_OFI_between_timestamps(df, start, end)`

Sum normalized OFI across a specified time interval. Where start and end are Pandas Timestamps. 

Parameters
----------
df : pandas.DataFrame
    Order book snapshot data.
start, end : label
    Interval slice on `df.index` (timestamps).

Returns
-------
pandas.Series
    Sum of each `ofi_m` between `start` and `end`.

Example
-------
```python
ofi_sum = compute_normalized_OFI_between_timestamps(df, '2024-10-21 11:54:29.221064336+00:00', '2024-10-21 11:54:29.764673165+00:00')
```  

---

### `compute_OFI_between_timestamps(df, start, end)`

Sum raw OFI across a specified time interval.

Parameters
----------
df : pandas.DataFrame
    Order book snapshot data.
start, end : label
    Interval slice on `df.index`.

Returns
-------
pandas.Series
    Sum of each `OFI_m` between `start` and `end`.

Example
-------
```python
ofi_sum = compute_OFI_between_timestamps(df, '09:30', '09:35')
```  

---

### `compute_integrated_OFI_between_timestamps(df, start, end, training_start, training_end)`

Compute integrated OFI over one interval by fitting PCA on another.

Parameters
----------
df : pandas.DataFrame
    Order book snapshot data.
start, end : label
    Interval for summing integrated OFI.
training_start, training_end : label
    Interval for fitting PCA weights on normalized OFI.

Returns
-------
integrated : pandas.Series
    Sum of integrated OFI between `start` and `end`.

weights : pandas.Series
    PCA loadings used for projection.

Example
-------
```python
integ_sum, w = compute_integrated_OFI_between_timestamps(
    df, 2024-10-21 11:54:29.221064336+00:00', '2024-10-21 11:54:29.764673165+00:00', training_start=0, training_end=1000)
```

---

### `compute_cross_asset_features(df_list, start, end, training_start, training_end, weights)`

Aggregate per-asset OFI features into a cross-asset summary.

Parameters
----------
df_list : list of pandas.DataFrame
    One DataFrame per asset, each indexed by timestamp.
start, end : label
    Common interval for summing normalized OFI per asset.
training_start, training_end : list of index
    PCA windows for each asset’s integrated OFI.
weights : list of float
    Scalar weights for combining each asset’s features.

Returns
-------
pandas.DataFrame
    Columns for each asset’s summed features and a final `COFI` column
    representing the weighted cross-asset OFI.

Example
-------
```python
cross_df = compute_cross_asset_features(
    [dfA, dfB], 2024-10-21 11:54:29.221064336+00:00', '2024-10-21 11:54:29.764673165+00:00', [0,0], [1000,1200], [0.5, 0.5]
)
```  

