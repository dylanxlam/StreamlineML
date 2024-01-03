# feature_interactions.py: streamlining the creation of interaction terms

# Feature interactions are created in machine learning to capture complex relationships between variables, 
    # enhancing predictive power and model flexibility by accounting for synergies and dependencies that may exist in the data.

# The `feature_interactions.py` module facilitates the creation and evaluation of interaction terms
    # in a DataFrame. It includes functions for generating polynomial features with interactions, creating 
    # custom interactions, performing statistical tests for interaction significance, evaluating 
    # mutual information, ANOVA, Kruskal-Wallis, Spearman correlation, and point-biserial correlation, 
    # identifying significant interactions, and creating time-lagged features, grouped statistics, and binned interactions.

# Example Module Import: from FeatEngToolkit import feature_interactions
# Example Function Import: from FeatEngToolkit.feature_interactions import polynomial_features_with_interactions

import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import ttest_ind, f_oneway, kruskal, pointbiserialr
import statsmodels.api as sm

def polynomial_features_with_interactions(df, degree=2, include_bias=False):
    """
    Create polynomial features up to the specified degree with interaction terms.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - degree (int): Degree of the polynomial features.
    - include_bias (bool): Whether to include a bias/intercept column.

    Returns:
    - pd.DataFrame: DataFrame with polynomial features and interactions.
    """
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias, interaction_only=True)
    poly_features = poly.fit_transform(df)
    poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names(df.columns))
    return poly_df

def generate_feature_crosses(df, feature_pairs):
    """
    Generate interaction terms by taking the cross product of specified feature pairs.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - feature_pairs (list of tuples): Pairs of features to create interaction terms.

    Returns:
    - pd.DataFrame: DataFrame with interaction terms.
    """
    crossed_df = df.copy()
    for feature_pair in feature_pairs:
        feature1, feature2 = feature_pair
        crossed_name = f"{feature1}_x_{feature2}"
        crossed_df[crossed_name] = df[feature1] * df[feature2]
    return crossed_df

def custom_interaction_terms(df, interaction_rules):
    """
    Generate custom interaction terms based on user-defined rules.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - interaction_rules (dict): Dictionary specifying interaction rules.

    Returns:
    - pd.DataFrame: DataFrame with custom interaction terms.
    """
    custom_df = df.copy()
    for interaction_name, rule in interaction_rules.items():
        custom_df[interaction_name] = rule(df)
    return custom_df

def statistical_interaction_tests(df, target_column, significance_threshold=0.05):
    """
    Perform statistical tests for the significance of interaction terms Using p-values from hypothesis tests

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - target_column (str): Name of the target column.
    - significance_threshold (float): Threshold for considering interactions significant.

    Returns:
    - pd.DataFrame: DataFrame with significant interaction terms.
    """
    
    interactions_df = pd.DataFrame(index=df.index)
    for feature_pair in combinations(df.columns, 2):
        interaction_name = f"{feature_pair[0]}_x_{feature_pair[1]}"
        p_value = evaluate_interaction_significance(df[feature_pair], df[target_column])
        if p_value < significance_threshold:
            interactions_df[interaction_name] = df[feature_pair[0]] * df[feature_pair[1]]
    return interactions_df

def evaluate_interaction_significance(df, feature_pair, target_column):
    """
    Perform a t-test to evaluate the significance of the interaction term.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - feature_pair (tuple): Pair of features for which the interaction is tested.
    - target_column (pd.Series): Target column.

    Returns:
    - float: p-value of the t-test.
    """
    feature1, feature2 = feature_pair
    group1 = target_column[df[feature1] == 0]
    group2 = target_column[df[feature2] == 1]

    _, p_value = ttest_ind(group1, group2)
    return p_value


def identify_significant_interactions(df, target_column, significance_threshold=0.05):
    """
    Identify significant interaction terms based on hypothesis tests.

    This function uses the OLS (Ordinary Least Squares) method from the statsmodels library 
        to perform hypothesis tests for each interaction term and includes the interaction term 
        in the result DataFrame if its p-value is below the specified threshold.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - target_column (str): The target column for hypothesis tests.
    - significance_threshold (float): The threshold for considering interactions significant.

    Returns:
    - pd.DataFrame: DataFrame with significant interaction terms.
    """
    interactions_df = pd.DataFrame(index=df.index)
    
    def evaluate_interaction_significance(feature_pair, target):
        X_interaction = sm.add_constant(feature_pair[0] * feature_pair[1])
        model = sm.OLS(target, X_interaction).fit()
        return model.pvalues[-1]

    for feature_pair in combinations(df.columns, 2):
        interaction_name = f"{feature_pair[0]}_x_{feature_pair[1]}"
        p_value = evaluate_interaction_significance(df[feature_pair], df[target_column])
        if p_value < significance_threshold:
            interactions_df[interaction_name] = df[feature_pair[0]] * df[feature_pair[1]]
    
    return interactions_df



def generate_polynomial_features(df, degree=2, columns=None):
    """
    Generate polynomial features up to a specified degree for selected columns.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - degree (int): Degree of the polynomial features.
    - columns (list): List of columns to apply polynomial features. If None, apply to all numeric columns.

    Returns:
    - pd.DataFrame: DataFrame with polynomial features.
    """
    if columns is None:
        columns = df.select_dtypes(include='number').columns

    poly = PolynomialFeatures(degree=degree)
    poly_features = poly.fit_transform(df[columns])
    poly_columns = poly.get_feature_names(columns)
    
    return pd.DataFrame(poly_features, columns=poly_columns, index=df.index)

def create_binned_interactions(df, bin_columns, interaction_columns):
    """
    Create interaction terms between binned features.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - bin_columns (list): List of columns to be binned.
    - interaction_columns (list): List of columns to create interactions between.

    Returns:
    - pd.DataFrame: DataFrame with binned interaction terms.
    """
    binned_df = df.copy()

    for column in bin_columns:
        binned_df[column + '_bin'] = pd.cut(df[column], bins=5, labels=False)

    interactions_df = pd.DataFrame(index=df.index)
    for feature_pair in combinations(binned_df.columns, 2):
        if feature_pair[0] in bin_columns and feature_pair[1] in interaction_columns:
            interaction_name = f"{feature_pair[0]}_x_{feature_pair[1]}"
            interactions_df[interaction_name] = binned_df[feature_pair[0]] * binned_df[feature_pair[1]]

    return interactions_df

from itertools import combinations
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def evaluate_chi2_interaction(df, target_column, columns):
    """
    Perform a chi-squared test to evaluate the significance of interaction terms.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - target_column (str): Name of the target column.
    - columns (list): List of columns to evaluate interactions.

    Returns:
    - pd.DataFrame: DataFrame with significant interaction terms based on chi-squared test.
    """
    interactions_df = pd.DataFrame(index=df.index)
    
    for feature_pair in combinations(columns, 2):
        chi2_stat, p_value, _, _ = chi2_contingency(pd.crosstab(df[feature_pair[0]], df[feature_pair[1]]))
        interaction_name = f"{feature_pair[0]}_x_{feature_pair[1]}"
        
        if p_value < 0.05:
            interactions_df[interaction_name] = df[feature_pair[0]] * df[feature_pair[1]]

    return interactions_df

def create_grouped_statistics(df, grouping_column, agg_functions, columns=None):
    """
    Create grouped statistics for specified columns based on a grouping column.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - grouping_column (str): Name of the column for grouping.
    - agg_functions (dict): Dictionary mapping column names to aggregation functions.
    - columns (list): List of columns to apply grouped statistics. If None, apply to all numeric columns.

    Returns:
    - pd.DataFrame: DataFrame with grouped statistics.
    """
    if columns is None:
        columns = df.select_dtypes(include='number').columns

    grouped_stats = df.groupby(grouping_column)[columns].agg(agg_functions)
    grouped_stats.columns = [f"{grouping_column}_{col}_{agg}" for col, agg in grouped_stats.columns]
    
    return grouped_stats.reset_index()

def create_time_lagged_features(df, time_column, value_column, max_lag=3):
    """
    Create time-lagged features for a given time series.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - time_column (str): Name of the column representing time.
    - value_column (str): Name of the column with values to be lagged.
    - max_lag (int): Maximum number of lags to create.

    Returns:
    - pd.DataFrame: DataFrame with time-lagged features.
    """
    time_lagged_df = df.copy()

    for lag in range(1, max_lag + 1):
        time_lagged_df[f"{value_column}_lag_{lag}"] = df[value_column].shift(lag)

    return time_lagged_df

from itertools import combinations
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression

def evaluate_mutual_information(df, target_column, columns=None):
    """
    Evaluate mutual information between features and target column.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - target_column (str): Name of the target column.
    - columns (list): List of columns to evaluate mutual information. If None, use all numeric columns.

    Returns:
    - pd.Series: Series with mutual information scores for each feature.
    """
    if columns is None:
        columns = df.select_dtypes(include='number').columns

    mutual_info_scores = df[columns].apply(lambda col: mutual_info_regression(col.values.reshape(-1, 1), df[target_column]))
    mutual_info_scores.index = columns

    return mutual_info_scores

def evaluate_anova(df, target_column, columns=None):
    """
    Perform one-way ANOVA to evaluate the significance of features.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - target_column (str): Name of the target column.
    - columns (list): List of columns to evaluate with ANOVA. If None, use all numeric columns.

    Returns:
    - pd.DataFrame: DataFrame with F-statistics and p-values for each feature.
    """
    if columns is None:
        columns = df.select_dtypes(include='number').columns

    anova_results = pd.DataFrame(index=columns, columns=['F-statistic', 'p-value'])

    for col in columns:
        groups = [group[col].dropna() for _, group in df.groupby(target_column)]
        f_statistic, p_value = f_oneway(*groups)
        anova_results.loc[col] = [f_statistic, p_value]

    return anova_results

def evaluate_kruskal_wallis(df, target_column, columns=None):
    """
    Perform Kruskal-Wallis H-test to evaluate the significance of features.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - target_column (str): Name of the target column.
    - columns (list): List of columns to evaluate with Kruskal-Wallis. If None, use all numeric columns.

    Returns:
    - pd.DataFrame: DataFrame with H-statistics and p-values for each feature.
    """
    if columns is None:
        columns = df.select_dtypes(include='number').columns

    kruskal_results = pd.DataFrame(index=columns, columns=['H-statistic', 'p-value'])

    for col in columns:
        groups = [group[col].dropna() for _, group in df.groupby(target_column)]
        h_statistic, p_value = kruskal(*groups)
        kruskal_results.loc[col] = [h_statistic, p_value]

    return kruskal_results

def evaluate_spearman_correlation(df, target_column, columns=None):
    """
    Evaluate Spearman rank correlation coefficients between features and target column.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - target_column (str): Name of the target column.
    - columns (list): List of columns to evaluate with Spearman correlation. If None, use all numeric columns.

    Returns:
    - pd.Series: Series with Spearman rank correlation coefficients for each feature.
    """
    if columns is None:
        columns = df.select_dtypes(include='number').columns

    spearman_correlations = df[columns].apply(lambda col: col.corr(df[target_column], method='spearman'))
    spearman_correlations.index = columns

    return spearman_correlations

def evaluate_point_biserial_correlation(df, binary_column, continuous_column):
    """
    Evaluate point-biserial correlation coefficient between binary and continuous columns.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - binary_column (str): Name of the binary column.
    - continuous_column (str): Name of the continuous column.

    Returns:
    - float: Point-biserial correlation coefficient.
    """
    x = df[continuous_column]
    y = df[binary_column]

    correlation, _ = pointbiserialr(x, y)
    return correlation
