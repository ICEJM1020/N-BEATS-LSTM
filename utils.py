import pandas as pd
import gzip
import pickle
import numpy as np
from scipy.stats import norm


APPS = [
    20, # microwave
    9, # kettle
    5, # washing machine
    34, # dish washer
    35, # oven
    42, # tumble dryer
    # 47, # hob & oven
    48, # hob
    28 # toaster
]

APPS_DICT = {
    20: 'microwave',
    9: 'kettle',
    5: 'washing machine',
    34: 'dish washer',
    35: 'oven',
    42: 'tumble dryer',
    # 47: 'hob & oven',
    48: 'hob',
    28: 'toaster'
}


def gap_lagu_extract(dataframe, lag_coef=0.8):
    df_len = dataframe.shape[0]
    app_df = dataframe[APPS]
    threshold = app_df.mean()

    feature_dic = {}

    for app in APPS:
        th=threshold[app]
        app_data = app_df[app]
        gap=[0]
        lagu=[th]
        for i in range(1, df_len):
            if app_data[i] < th:
                gap.append(gap[-1] + 1)
                lagu.append(lagu[-1] * lag_coef)
            else:
                gap.append(0)
                lagu.append(app_data[i])

        feature_dic['gap_'+str(app)]=gap
        feature_dic['lagu_'+str(app)]=lagu

    return pd.DataFrame(feature_dic)


def _normal_probability(std):
    timestamp = np.linspace(-3, 3, 2*std)
    _norm_h = norm.cdf(timestamp[:std+1])
    _norm = [_norm_h[0]]
    for i in range(1, 2*std+1):
        if i<=std:
            _norm.append(_norm_h[i]-_norm_h[i-1])
        else:
            _norm.append(_norm[2*std-i])
    return np.array(_norm)

def normalize_data(dataframe, std=5):
    norm = _normal_probability(std)
    th_list = dataframe.describe().loc['50%']
    for col_name in APPS:
        th = float(th_list[col_name])
        col_data = dataframe[col_name].to_numpy()
        new_data = np.ones_like(col_data)
        new_data = new_data * float(th)
        length = col_data.shape[0]
        for i in range(length):
            if col_data[i] > th:
                start = 0 if i-std<0 else i-std
                end = length if i+std+1>length else i+std+1
                norm_multoplier = norm[int(std-(i-start)) : int(std + (end-i))]
                new_data[start:end] = new_data[start:end] + norm_multoplier * col_data[i]
        dataframe['norm_'+str(col_name)] = new_data
    return dataframe


def consumption_count(data):
    th = data.mean()
    x = []
    for i, value in enumerate(data):
        if value >= th:
            x.append(value)
    x = np.array(x)
    x.sort()
    return x.mean(), x.std()

def get_consum_mean_std(df):
    output = {}
    for index, col in df.iteritems():
        static = consumption_count(col)
        output[index] = {'mean':static[0], 'std':static[1]}
    return pd.DataFrame(output)


def get_dataframe(data_file_path, scale=2, shift=-1, normstd=7):
    ## load the data file
    with gzip.open(data_file_path, 'rb') as data_file:
        data = pickle.load(data_file)
    df = pd.DataFrame(data)

    if normstd > 0 :
        df = normalize_data(df, std=normstd)

    ## Standardization
    train_min = df.describe().loc['75%']
    train_max = df.max()
    df_std = ((df - train_min) / (train_max-train_min)) * scale + shift

    return df_std

