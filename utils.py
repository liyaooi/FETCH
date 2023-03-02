import logging
import os
import sys
import time

import numpy as np


def log_dir(exp_dir):
    """Make directory of log file and initialize logging module"""
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    log_format = '%(asctime)s %(filename)s:%(lineno)d %(funcName)s %(levelname)s | %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, force=True)
    fh = logging.FileHandler(os.path.join(exp_dir, "log.txt"))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)


def get_key_from_dict(d, value):
    k = [k for k, v in d.items() if v == value]
    return k


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    logging.info('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    logging.info('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    logging.info('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
