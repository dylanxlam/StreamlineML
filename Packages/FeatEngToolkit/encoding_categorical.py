# encoding_categorical.py: Enhancing Categorical Variable Encoding

# The encoding_categorical module in StreamlineML's Feature Engineering 
    # Toolkit provides a set of functions for advanced encoding of 
    # categorical variables. Built on top of pandas, numpy, and scikit-learn, 
    # these functions offer encoding techniques not included in the standard 
    # libraries. From one-hot encoding and target encoding to custom and 
    # novel methods like Levenshtein and Weight of Evidence encoding, 
    # this module empowers users to handle categorical data with flexibility and precision. 
    # Each function is designed to seamlessly integrate into data preprocessing workflows, 
    # contributing to the robustness and effectiveness of machine learning models.

# Example Module Import: from FeatEngToolkit import encoding_categorical
# Example Function Import: from FeatEngToolkit.encoding_categorical import one_hot_encoding


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import FeatureHasher


def one_hot_encoding(df, columns=None):
    """
    Perform one-hot encoding on specified columns or all categorical columns.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - columns (list): List of columns to one-hot encode.

    Returns:
    - pd.DataFrame: DataFrame with one-hot encoded columns.
    """
    if columns is None:
        columns = df.select_dtypes(include='object').columns

    df = pd.get_dummies(df, columns=columns, drop_first=True)
    return df

def target_encoding(df, target_column, columns=None):
    """
    Perform target encoding on specified columns or all categorical columns.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - target_column (str): The target column for encoding.
    - columns (list): List of columns to target encode.

    Returns:
    - pd.DataFrame: DataFrame with target encoded columns.
    """
    if columns is None:
        columns = df.select_dtypes(include='object').columns

    for col in columns:
        mapping = df.groupby(col)[target_column].mean()
        df[col] = df[col].map(mapping)

    return df

def frequency_encoding(df, columns=None):
    """
    Perform frequency encoding on specified columns or all categorical columns.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - columns (list): List of columns to frequency encode.

    Returns:
    - pd.DataFrame: DataFrame with frequency encoded columns.
    """
    if columns is None:
        columns = df.select_dtypes(include='object').columns

    for col in columns:
        mapping = df[col].value_counts(normalize=True)
        df[col] = df[col].map(mapping)

    return df

def label_encoding(df, columns=None):
    """
    Perform label encoding on specified columns or all categorical columns.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - columns (list): List of columns to label encode.

    Returns:
    - pd.DataFrame: DataFrame with label encoded columns.
    """
    if columns is None:
        columns = df.select_dtypes(include='object').columns

    le = LabelEncoder()
    for col in columns:
        df[col] = le.fit_transform(df[col])

    return df

def levenshtein_encoding(df, column):
    """
    Perform Levenshtein encoding for a categorical column.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column (str): Name of the categorical column to be encoded.

    Returns:
    - pd.DataFrame: DataFrame with the Levenshtein-encoded column added.
    """
    levenshtein_dict = {}
    for category in df[column].unique():
        distances = [levenshtein_distance(category, other) for other in df[column].unique()]
        levenshtein_dict[category] = np.mean(distances)
    
    df[column + '_levenshtein'] = df[column].map(levenshtein_dict)
    return df

def levenshtein_distance(s1, s2):
    """
    Compute the Levenshtein distance between two strings.

    Parameters:
    - s1 (str): First string.
    - s2 (str): Second string.

    Returns:
    - int: Levenshtein distance between the two strings.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))

        previous_row = current_row

    return previous_row[-1]
def weight_of_evidence_encoding(df, column, target_column):
    """
    Perform Weight of Evidence encoding for a categorical column.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column (str): Name of the categorical column to be encoded.
    - target_column (str): Name of the target column.

    Returns:
    - pd.DataFrame: DataFrame with the Weight of Evidence-encoded column added.
    """
    category_counts = df.groupby(column)[target_column].agg(['count', 'sum'])
    total_count = df[target_column].count()
    total_sum = df[target_column].sum()

    woe_dict = {}
    for category in df[column].unique():
        category_count = category_counts.loc[category, 'count']
        category_sum = category_counts.loc[category, 'sum']
        non_category_count = total_count - category_count
        non_category_sum = total_sum - category_sum

        woe = np.log((category_sum / category_count) / (non_category_sum / non_category_count))
        woe_dict[category] = woe

    df[column + '_woe'] = df[column].map(woe_dict)
    return df

def ordinal_encoding(df, mapping_dict):
    """
    Perform ordinal encoding on specified columns using a provided mapping.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - mapping_dict (dict): Dictionary mapping categorical values to numerical values for each column.

    Returns:
    - pd.DataFrame: DataFrame with ordinal encoded columns.
    """
    df_copy = df.copy()
    for col, mapping in mapping_dict.items():
        df_copy[col] = df_copy[col].map(mapping)
    return df_copy

def binary_encoding(df, columns=None):
    """
    Perform binary encoding on specified columns or all categorical columns.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - columns (list): List of columns to binary encode.

    Returns:
    - pd.DataFrame: DataFrame with binary encoded columns.
    """
    if columns is None:
        columns = df.select_dtypes(include='object').columns

    df_copy = df.copy()
    for col in columns:
        binary_encoded = pd.get_dummies(df_copy[col], prefix=col, drop_first=True)
        df_copy = pd.concat([df_copy, binary_encoded], axis=1)
        df_copy.drop(col, axis=1, inplace=True)

    return df_copy

def hashing_encoding(df, columns=None, n_components=8):
    """
    Perform hashing encoding on specified columns or all categorical columns.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - columns (list): List of columns to hash encode.
    - n_components (int): Number of components for hashing.

    Returns:
    - pd.DataFrame: DataFrame with hashed encoded columns.
    """
    if columns is None:
        columns = df.select_dtypes(include='object').columns

    df_copy = df.copy()
    hasher = FeatureHasher(n_features=n_components, input_type='string')
    for col in columns:
        hashed_features = hasher.transform(df_copy[[col]].astype(str).values).toarray()
        hashed_df = pd.DataFrame(hashed_features, columns=[f'{col}_hash_{i}' for i in range(n_components)])
        df_copy = pd.concat([df_copy, hashed_df], axis=1)
        df_copy.drop(col, axis=1, inplace=True)

    return df_copy

def custom_encoding(df, encoding_func, columns=None, **kwargs):
    """
    Perform custom encoding using a user-defined encoding function on specified columns or all categorical columns.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - encoding_func (callable): Custom encoding function.
    - columns (list): List of columns to custom encode.
    - **kwargs: Additional arguments for the custom encoding function.

    Returns:
    - pd.DataFrame: DataFrame with custom encoded columns.
    """
    if columns is None:
        columns = df.select_dtypes(include='object').columns

    df_copy = df.copy()
    for col in columns:
        encoded_col = encoding_func(df_copy[col], **kwargs)
        df_copy[col] = encoded_col

    return df_copy

def woe_encoding(df, target_column, columns=None):
    """
    Perform Weight of Evidence encoding on specified columns or all categorical columns.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - target_column (str): The target column for encoding.
    - columns (list): List of columns to WoE encode.

    Returns:
    - pd.DataFrame: DataFrame with WoE encoded columns.
    """
    if columns is None:
        columns = df.select_dtypes(include='object').columns

    for col in columns:
        grouped_data = df.groupby(col)[target_column].agg(['count', 'sum'])
        total_count = grouped_data['count'].sum()
        total_positive = grouped_data['sum'].sum()
        woe_mapping = np.log((grouped_data['sum'] / total_positive) / ((grouped_data['count'] - grouped_data['sum']) / (total_count - total_positive)))
        df[col] = df[col].map(woe_mapping)

    return df

def probability_ratio_encoding(df, target_column, columns=None):
    """
    Perform Probability Ratio Encoding on specified columns or all categorical columns.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - target_column (str): The target column for encoding.
    - columns (list): List of columns to encode.

    Returns:
    - pd.DataFrame: DataFrame with Probability Ratio encoded columns.
    """
    if columns is None:
        columns = df.select_dtypes(include='object').columns

    for col in columns:
        grouped_data = df.groupby(col)[target_column].agg(['count', 'sum'])
        non_target_count = grouped_data['count'] - grouped_data['sum']
        target_count = grouped_data['sum']
        probability_ratio_mapping = target_count / non_target_count
        df[col] = df[col].map(probability_ratio_mapping)

    return df
