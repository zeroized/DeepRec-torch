import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.cluster import KMeans

from feature.feature_meta import FeatureMeta
from util.log_util import *
from preprocess.discretize import *


@DeprecationWarning
def get_idx_and_value(feat_meta: FeatureMeta, raw_data: pd.DataFrame):
    logger = create_console_logger(name='feat_meta')
    write_info_log(logger, 'preprocess started')
    idx = 0
    # allocate indices for continuous features
    continuous_feats = feat_meta.continuous_feats
    for name in continuous_feats:
        continuous_feat = continuous_feats[name]
        continuous_feat.start_idx = idx
        idx += 1
    # generate label encoders and allocate indices range for categorical features
    categorical_feats = feat_meta.categorical_feats
    for name in categorical_feats:
        categorical_feat = categorical_feats[name]
        le = categorical_feat.processor
        if le:
            num_classes = len(le.classes_)
            raw_data[name] = le.transform(raw_data[name])
        else:
            le = LabelEncoder()
            le.fit(raw_data[name])
            categorical_feat.processor = le
            num_classes = len(le.classes_)
            categorical_feat.dim = num_classes
        categorical_feat.start_idx = idx
        idx += num_classes
    # generate multi-hot encoders and allocate indices range for multi-category features
    multi_category_feats = feat_meta.multi_category_feats
    for name in multi_category_feats:
        multi_category_feat = multi_category_feats[name]
        le = multi_category_feat.processor
        if le:
            num_classes = len(le.classes_)
        else:
            mlb = MultiLabelBinarizer()
            mlb.fit(raw_data[name])
            multi_category_feat.processor = mlb
            num_classes = len(mlb.classes_)
            multi_category_feat.dim = num_classes
        multi_category_feat.start_idx = idx
        idx += num_classes
    write_info_log(logger, 'feature meta updated')
    # transform raw data to index and value form
    write_info_log(logger, 'index and value transformation started')
    feat_df = raw_data.apply(process_line, feat_meta=feat_meta, axis=1)
    write_info_log(logger, 'preprocess finished')
    return feat_df.feat_idx.values.tolist(), feat_df.feat_value.values.tolist()


def preprocess_features(feat_meta: FeatureMeta, data: pd.DataFrame, split_continuous_category=False):
    r"""Transform raw data into index and value form.
    Continuous features will be discretized, standardized, normalized or scaled according to feature meta.
    Categorical features will be encoded with a label encoder.


    :param feat_meta: The FeatureMeta instance that describes raw_data.
    :param data: The raw_data to be transformed.
    :param split_continuous_category: Whether to return value of continuous features and index of category features.
    :return: feat_index, feat_value, category_index, continuous_value
    """
    logger = create_console_logger(name='feat_meta')
    write_info_log(logger, 'preprocess started')

    idx = 0
    continuous_feats = feat_meta.continuous_feats
    categorical_feats = feat_meta.categorical_feats
    columns = list(continuous_feats.keys())
    columns.extend(list(categorical_feats.keys()))
    data = data[columns]
    feat_idx = pd.DataFrame()

    # transform continuous features
    write_info_log(logger, 'transforming continuous features')
    feat_value_continuous = pd.DataFrame()
    for name in continuous_feats:
        feat = continuous_feats[name]
        feat.start_idx = idx
        if not feat.discretize:
            # standardized, normalize or scale
            processor = feat.transformation
            col_data = np.reshape(data[name].values, (-1, 1))
            col_data = processor.fit_transform(col_data)
            col_data = np.reshape(col_data, -1)
            feat_value_continuous[name] = col_data
            feat_idx[name] = np.repeat(idx, repeats=len(data))
            idx += 1
        else:
            # discretize
            discrete_data, intervals = discretize(data[name], feat.discretize, feat.dim)
            feat.bins = intervals
            feat_idx[name] = discrete_data + idx
            feat_value_continuous[name] = pd.Series(np.ones(len(data[name])))
            idx += feat.dim

    write_info_log(logger, 'transforming categorical features')
    # transform categorical features
    category_index = pd.DataFrame()
    for name in categorical_feats:
        categorical_feat = categorical_feats[name]
        le = LabelEncoder()
        feat_idx[name] = le.fit_transform(data[name]) + idx
        category_index[name] = feat_idx[name]
        categorical_feat.processor = le
        num_classes = len(le.classes_)
        categorical_feat.dim = num_classes
        categorical_feat.start_idx = idx
        idx += num_classes

    # TODO add multi category features
    feat_idx = feat_idx.apply(lambda x: x.values, axis=1)
    category_index = category_index.apply(lambda x: x.values, axis=1)

    feat_value_category = pd.DataFrame(np.ones((len(data), len(categorical_feats.keys()))))
    feat_value = pd.concat([feat_value_continuous, feat_value_category], axis=1)
    feat_value = feat_value.apply(lambda x: x.values, axis=1)
    continuous_value = feat_value_continuous.apply(lambda x: x.values, axis=1)

    write_info_log(logger, 'preprocess finished')
    if split_continuous_category:
        return feat_idx, feat_value, category_index, continuous_value
    return feat_idx, feat_value


def process_line(row, feat_meta):
    feat_idx, feat_value = [], []
    # process continuous features
    continuous_feats = feat_meta.continuous_feats
    for feat_name in continuous_feats:
        feat = continuous_feats[feat_name]
        row_value = row[feat_name]
        idx, value = feat.get_idx_and_value(row_value)
        feat_idx.append(idx)
        feat_value.append(value)
    # process categorical features
    categorical_feats = feat_meta.categorical_feats
    for feat_name in categorical_feats:
        feat = categorical_feats[feat_name]
        row_value = row[feat_name]
        idx, value = feat.get_idx_and_value(row_value)
        feat_idx.append(idx)
        feat_value.append(value)
    # process multi-category features
    multi_category_feats = feat_meta.multi_category_feats
    for feat_name in multi_category_feats:
        feat = multi_category_feats[feat_name]
        row_value = row[feat_name]
        idxes, values = feat.get_idx_and_value(row_value)
        feat_idx.extend(idxes)
        feat_value.extend(values)
    return pd.Series(index=['feat_idx', 'feat_value'], data=[feat_idx, feat_value])

