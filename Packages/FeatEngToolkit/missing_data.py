# missing_data.py: Advanced Missing Data Handling for Pandas

# A Module designed to enhance Pandas' missing data handling capabilities. 
    # Enhance Pandas' missing data capabilities with this comprehensive toolkit. 
    # Designed on top of the Pandas library, it introduces advanced imputation strategies, 
    # categorical data handling, and more. Simplify your data preprocessing workflows for 
    # efficient missing data handling. Impute missing values with KNN, statistical methods, 
    # and time series imputation. Explore, visualize, and assess missing data quality 
    # interactively with added functionalities. Elevate your Pandas DataFrames' missing 
    # data handling with this versatile module.

# Example Module Import: from FeatEngToolkit import missing_data
# Example Function Import: from FeatEngToolkit.missing_data import handle_missing_values

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

def handle_missing_values(df, strategy='knn', k_neighbors=5):
    """
    Handle missing values in a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - strategy (str): Strategy to handle missing values ('knn', 'mean', 'median', 'mode', 'drop').
    - k_neighbors (int): Number of neighbors for KNN imputation.

    Returns:
    - pd.DataFrame: DataFrame with missing values handled.
    """
    if strategy == 'drop':
        return df.dropna()

    if strategy == 'mean':
        return df.fillna(df.mean())

    if strategy == 'median':
        return df.fillna(df.median())

    if strategy == 'mode':
        return df.fillna(df.mode().iloc[0])

    if strategy == 'knn':
        return knn_imputation(df, k_neighbors)

    raise ValueError("Invalid strategy. Choose from 'knn', 'mean', 'median', 'mode', or 'drop'.")

def knn_imputation(df, k_neighbors):
    """
    KNN imputation (K-nearest neighbor model):
        A new sample is imputed by finding the samples in the training set 
        “closest” to it and averages these nearby points to fill in the value.

    Impute missing values using KNN imputation.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - k_neighbors (int): Number of neighbors for KNN imputation.

    Returns:
    - pd.DataFrame: DataFrame with missing values imputed using KNN.
    """
    imputer = KNNImputer(n_neighbors=k_neighbors)
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

def impute_categorical_missing(df, strategy='most_frequent'):
    """
    Impute missing values in categorical columns.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - strategy (str): Strategy for imputing categorical missing values ('most_frequent', 'constant', etc.).

    Returns:
    - pd.DataFrame: DataFrame with missing values in categorical columns imputed.
    """
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].value_counts().idxmax() if strategy == 'most_frequent' else 'Unknown')
    return df


def impute_numerical_missing(df, strategy='mean'):
    """
    Impute missing values in numerical columns.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - strategy (str): Strategy for imputing numerical missing values ('mean', 'median', 'constant').

    Returns:
    - pd.DataFrame: DataFrame with missing values in numerical columns imputed.
    """
    numerical_cols = df.select_dtypes(include=['number']).columns
    imputer = SimpleImputer(strategy=strategy)
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
    return df

def impute_rf(df):
    """
    Impute missing values using RandomForestRegressor.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with missing values.

    Returns:
    - pd.DataFrame: DataFrame with missing values imputed.
    """
    # Assuming 'target_column' is the column with missing values
    target_column = 'target_column'
    
    # Split data into complete and incomplete sets
    complete_data = df.dropna(subset=[target_column])
    incomplete_data = df[df[target_column].isnull()]

    # Use RandomForestRegressor to impute missing values
    rf_imputer = RandomForestRegressor()
    rf_imputer.fit(complete_data.drop(target_column, axis=1), complete_data[target_column])
    
    # Predict missing values
    imputed_values = rf_imputer.predict(incomplete_data.drop(target_column, axis=1))
    
    # Update the DataFrame with imputed values
    df.loc[df[target_column].isnull(), target_column] = imputed_values

    return df

def impute_time_series(df, time_column, value_column):
    """
    Impute missing values in a time series using linear interpolation.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with missing values.
    - time_column (str): The column representing time in the DataFrame.
    - value_column (str): The column with missing values to be imputed.

    Returns:
    - pd.DataFrame: DataFrame with missing values imputed using linear interpolation.
    """
    # Assuming the DataFrame is sorted by the time column
    df[value_column].interpolate(method='time', inplace=True)
    return df

def handle_outliers_zscore(df, columns, threshold=3):
    """
    Handle outliers in specified columns using the Z-score method.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - columns (list): List of columns to handle outliers.
    - threshold (float): Z-score threshold for identifying outliers.

    Returns:
    - pd.DataFrame: DataFrame with outliers handled.
    """
    df_no_outliers = df.copy()

    for column in columns:
        z_scores = (df_no_outliers[column] - df_no_outliers[column].mean()) / df_no_outliers[column].std()
        outliers_mask = (z_scores.abs() > threshold)
        df_no_outliers.loc[outliers_mask, column] = df_no_outliers[column].median()

    return df_no_outliers

def explore_missing_data(df):
    """
    Explore missing data patterns interactively.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - None (displays interactive visualizations).
    - Suggested: use the 'pattern_imputation' function to impute missing values based on observed patterns
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False)
    plt.title('Missing Data Exploration')
    plt.show()

def ensemble_imputation(df, strategies=('mean', 'median', 'knn')):
    """
    Implement ensemble imputation methods.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - strategies (tuple): Imputation strategies to combine.

    Returns:
    - pd.DataFrame: DataFrame with missing values imputed using ensemble methods.
    """
    imputed_dfs = []

    for strategy in strategies:
        if strategy == 'knn':
            # KNN imputation
            imputer = IterativeImputer(estimator='knn', random_state=42)
        else:
            # Other strategies (mean, median, etc.)
            imputer = SimpleImputer(strategy=strategy)

        imputed_data = imputer.fit_transform(df)
        imputed_dfs.append(pd.DataFrame(imputed_data, columns=df.columns))

    # Combine imputed DataFrames
    df_imputed = pd.concat(imputed_dfs, keys=strategies, axis=1)
    
    return df_imputed

def visualize_missing_data(df):
    """
    Visualize the distribution of missing values in a dataset.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - None (displays visualizations).
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False)
    plt.title('Missing Data Visualization')
    plt.show()

def augment_dataset(df, fraction=0.1, seed=None):
    """
    Augment a dataset by introducing synthetic missing values.

    Parameters:
    - df (pd.DataFrame): The original DataFrame.
    - fraction (float): The fraction of entries to be replaced with NaN.
    - seed (int): Seed for reproducibility.

    Returns:
    - pd.DataFrame: Augmented DataFrame with synthetic missing values.
    """
    np.random.seed(seed)
    augmented_df = df.copy()

    # Randomly select a fraction of entries to be replaced with NaN
    mask = np.random.rand(*df.shape) < fraction
    augmented_df[mask] = np.nan

    return augmented_df

def imputation_quality_metrics(original_df, imputed_df):
    """
    Evaluate the quality of imputations using Mean Squared Error.

    Parameters:
    - original_df (pd.DataFrame): The original DataFrame.
    - imputed_df (pd.DataFrame): The DataFrame with imputed values.

    Returns:
    - float: Mean Squared Error between original and imputed values.
    """
    # Assuming both dataframes have the same shape and indices
    mse = mean_squared_error(original_df.values.flatten(), imputed_df.values.flatten())
    return mse

def interactive_imputation(df, indices_to_impute, imputation_values):
    """
    Allow users to interactively input or validate imputed values.

    Parameters:
    - df (pd.DataFrame): The original DataFrame.
    - indices_to_impute (list or np.ndarray): Indices to impute interactively.
    - imputation_values (list or np.ndarray): Values for interactive imputation.

    Returns:
    - pd.DataFrame: DataFrame with interactively imputed values.
    """
    interactive_df = df.copy()
    interactive_df.iloc[indices_to_impute] = imputation_values
    return interactive_df

def pattern_imputation(df):
    """
    Impute missing values based on observed patterns in the dataset.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with missing values.

    Returns:
    - pd.DataFrame: DataFrame with missing values imputed based on observed patterns.
    """
    # Identify columns with missing values
    columns_with_missing = df.columns[df.isnull().any()].tolist()

    for column in columns_with_missing:
        # Find columns that often have missing values together with the current column
        correlated_columns = df.corr().loc[:, column]
        correlated_columns = correlated_columns[correlated_columns > 0.5].index.tolist()

        # Check if there are correlated columns with missing values
        correlated_columns_with_missing = [col for col in correlated_columns if col in columns_with_missing]

        if correlated_columns_with_missing:
            # Impute missing values based on the mean of correlated columns
            df[column].fillna(df[correlated_columns_with_missing].mean(axis=1), inplace=True)

    return df
