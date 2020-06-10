import pandas as pd
from feature.feature_meta import FeatureMeta
from preprocess.feat_engineering import preprocess_features


def load_data(path, n_samples=-1):
    raw_ratings = pd.read_csv(path)
    if n_samples > 0:
        raw_ratings = raw_ratings.sample(n=n_samples)
        raw_ratings.reset_index(drop=True, inplace=True)

    return raw_ratings


def load_and_preprocess(path, n_samples=10000, binarize_label=True):
    raw_ratings = load_data(path, n_samples)

    feature_meta = FeatureMeta()
    feature_meta.add_categorical_feat('userId')
    feature_meta.add_categorical_feat('movieId')

    X = raw_ratings
    X_idx, X_value = preprocess_features(feature_meta, X)
    y = raw_ratings.rating

    if binarize_label:
        def transform_y(label):
            if label > 3:
                return 1
            else:
                return 0

        y = y.apply(transform_y)
    return X_idx, X_value, y, feature_meta
