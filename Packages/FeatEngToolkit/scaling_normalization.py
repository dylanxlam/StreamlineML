# Functions not available in libraries like scikit-learn

from scipy.stats import yeojohnson, boxcox
from sklearn.preprocessing import normalize
import numpy as np

def boxcox_transformation(df, columns=None):
    """
    Perform Box-Cox transformation on specified columns or all numeric columns.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - columns (list): List of columns to Box-Cox transform. If None, transform all numeric columns.

    Returns:
    - pd.DataFrame: DataFrame with Box-Cox transformed columns.
    """
    if columns is None:
        columns = df.select_dtypes(include='number').columns

    for col in columns:
        df[col], _ = boxcox(df[col] + 1)  # Adding 1 to handle zero values

    return df

def yeojohnson_transformation(df, columns=None):
    """
    Perform Yeo-Johnson transformation on specified columns or all numeric columns.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - columns (list): List of columns to Yeo-Johnson transform. If None, transform all numeric columns.

    Returns:
    - pd.DataFrame: DataFrame with Yeo-Johnson transformed columns.
    """
    if columns is None:
        columns = df.select_dtypes(include='number').columns

    for col in columns:
        df[col], _ = yeojohnson(df[col] + 1)  # Adding 1 to handle zero values

    return df

def rank_transformation(df, columns=None):
    """
    Perform rank transformation on specified columns or all numeric columns.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - columns (list): List of columns to rank transform. If None, transform all numeric columns.

    Returns:
    - pd.DataFrame: DataFrame with rank-transformed columns.
    """
    if columns is None:
        columns = df.select_dtypes(include='number').columns

    for col in columns:
        df[col] = df[col].rank()

    return df

def quantile_transformation(df, columns=None):
    """
    Perform quantile transformation on specified columns or all numeric columns.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - columns (list): List of columns to quantile transform. If None, transform all numeric columns.

    Returns:
    - pd.DataFrame: DataFrame with quantile-transformed columns.
    """
    if columns is None:
        columns = df.select_dtypes(include='number').columns

    for col in columns:
        df[col] = np.percentile(df[col], np.linspace(0, 100, len(df)))

    return df


def pareto_scaling(df, columns=None):
    """
    Perform Pareto scaling on specified columns or all numeric columns.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - columns (list): List of columns to Pareto scale. If None, scale all numeric columns.

    Returns:
    - pd.DataFrame: DataFrame with Pareto-scaled columns.
    """
    if columns is None:
        columns = df.select_dtypes(include='number').columns

    for col in columns:
        df[col] = df[col] / np.sqrt(df[col].std())

    return df


def unit_vector_transformation(df, columns=None):
    """
    Perform unit vector transformation on specified columns or all numeric columns.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - columns (list): List of columns to transform. If None, transform all numeric columns.

    Returns:
    - pd.DataFrame: DataFrame with unit vector-transformed columns.
    """
    if columns is None:
        columns = df.select_dtypes(include='number').columns

    df[columns] = normalize(df[columns], axis=0)

    return df

def power_transformation(df, power=2, columns=None):
    """
    Perform power transformation on specified columns or all numeric columns.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - power (float): Power value for transformation.
    - columns (list): List of columns to transform. If None, transform all numeric columns.

    Returns:
    - pd.DataFrame: DataFrame with power-transformed columns.
    """
    if columns is None:
        columns = df.select_dtypes(include='number').columns

    df[columns] = df[columns] ** power

    return df

