from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import feature_generate_memory, utils_memory


def label_encode_to_onehot(data, data_train=None):
    """
    :param x:np.array
    :return:one_hot
    """
    onehot = None
    data = data.astype(int)
    for col in range(data.shape[1]):
        if isinstance(data_train, np.ndarray):
            x = data[:, col]
            x_ = data_train[:, col]
            num_class = int(np.max(x_) + 1)
        else:
            x = data[:, col]
            num_class = int(np.max(x) + 1)
        m = np.zeros((len(x), num_class))
        m[range(len(x)), x] = 1
        if not isinstance(onehot, np.ndarray):
            onehot = m
        else:
            onehot = np.concatenate((onehot, m), axis=1)

    return onehot


def split_train_test(df, d_columns, target, mode, train_size, seed, shuffle):
    """
    Split data into training set and test set

    :param df: pd.DataFrame, origin data
    :param d_columns: a list of the names of discrete columns
    :param target: str, label name
    :param mode: str, classify or regression
    :param seed: int, to fix random seed
    :param train_size: float
    :return: df_train_val, df_test
    """
    # for col in d_columns:
    #     new_fe = merge_categories(df[col].values)
    #     df[col] = new_fe

    if mode == "classify":
        df_train_val, df_test = train_test_split(df, train_size=train_size, random_state=seed,
                                                 stratify=df[target], shuffle=shuffle)
    else:
        df_train_val, df_test = train_test_split(df, train_size=train_size, random_state=seed, shuffle=shuffle)

    # df_train_val = df_train_val.copy()
    # for col in d_columns:
    #     new_fe = merge_categories(df_train_val[col].values)
    #     df_train_val[col] = new_fe

    return df_train_val, df_test


def merge_categories(col, threshold=0.001):
    """
    Remove the categories with less than 'nums' occurrences in the discrete features,
    and replace them with the categories with the fewest occurrences.

    :param col: pd.DataFrame, one column
    :param threshold: threshold
    :return: pd.DataFrame, the column after merging
    """
    nums = max(int(len(col) * threshold), 5)
    count = dict(Counter(col))
    sorted_count = dict(sorted(count.items(), key=lambda x: x[1], reverse=True))
    # print(sorted_count)
    replace_dict = {}
    replace = False
    merge_name = None
    for name in sorted_count.keys():
        if replace:
            replace_dict[name] = merge_name
        else:
            if sorted_count[name] < nums:
                merge_name = name
                replace = True
    new_fe = [replace_dict[x] if x in replace_dict else x for x in col]
    return new_fe


def features_process(df_train_val, mode, c_columns, d_columns, target):
    # 分类任务的label数值化
    df_train_val.reset_index(drop=True, inplace=True)
    if mode == "classify":
        col = df_train_val[target].values.reshape(-1)
        df_train_val[target] = utils_memory.categories_to_int(col)
    df_t = df_train_val[target]
    df_t_norm = pd.DataFrame({target: feature_generate_memory.normalization(df_t.values).reshape(-1)})

    # 连续数据编码
    df_c_encode = pd.DataFrame()
    if len(c_columns):
        for col in c_columns:
            df_c_encode[col] = feature_generate_memory.normalization(df_train_val[col].values).reshape(-1)
        df_c_encode = pd.concat((df_c_encode, df_t_norm), axis=1)

    else:
        df_c_encode = pd.DataFrame()

    df_d_labelencode = pd.DataFrame()
    for column in d_columns:
        df_d_labelencode[column] = utils_memory.categories_to_int(df_train_val[column].values)

    df_d_encode = pd.DataFrame()
    for col in df_train_val.columns:
        if col in c_columns:
            fe = feature_generate_memory.binning_with_tree(df_train_val[col].values, df_t.values)
            df_d_encode[col] = feature_generate_memory.normalization(fe).reshape(-1)
        if col in d_columns:
            df_d_encode[col] = feature_generate_memory.normalization(df_d_labelencode[col].values).reshape(-1)
    df_d_encode = pd.concat((df_d_encode, df_t_norm), axis=1)

    return df_d_labelencode, df_c_encode, df_d_encode, df_t, df_t_norm


def remove_duplication(data):
    """
    Remove duplicated columns

    :param data: pd.DataFrame
    :return: pd.DataFrame or np.array, sorted index of duplicated columns
    """
    _, idx = np.unique(data, axis=1, return_index=True)
    y = data.iloc[:, np.sort(idx)]
    return y, np.sort(idx)
