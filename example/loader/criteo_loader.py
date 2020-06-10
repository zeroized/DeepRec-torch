import pandas as pd
from feature.feature_meta import FeatureMeta
from preprocess.feat_engineering import preprocess_features

continuous_columns = ['I' + str(i) for i in range(1, 14)]
category_columns = ['C' + str(i) for i in range(1, 27)]
columns = ['y']


def load_data(path, n_samples=-1):
    data = pd.read_csv(path, delimiter='\t', header=None)

    columns.extend(continuous_columns)
    columns.extend(category_columns)
    data.columns = columns

    if n_samples > 0:
        data = data.sample(n=n_samples)
        data.reset_index(drop=True, inplace=True)
    return data


def missing_values_process(data):
    continuous_fillna = data[continuous_columns].fillna(data[continuous_columns].median())
    data[continuous_columns] = continuous_fillna
    category_fillna = data[category_columns].fillna('0')
    data[category_columns] = category_fillna
    return data


def load_and_preprocess(path, n_samples=10000, discretize=False):
    r""" An example for load and preprocess criteo dataset

    :param path: File path of criteo dataset.
    :param n_samples: Number to sample from the full dataset. n_samples <= 0 means not to sample.
    :param discretize: Whether to discretize continuous features. All features will be processed with the same method
    :return: X_idx,X_value,y,feature_meta
    """
    # discretize in {False,'linear','non-linear'}, if non-linear
    data = load_data(path, n_samples=n_samples)

    # fill NaN
    data = missing_values_process(data)

    # build feature meta instance
    feature_meta = FeatureMeta()
    for column in continuous_columns:
        feature_meta.add_continuous_feat(column, discretize=discretize)
    for column in category_columns:
        feature_meta.add_categorical_feat(column)

    X_idx, X_value = preprocess_features(feature_meta, data)
    y = data.y
    return X_idx, X_value, y, feature_meta
