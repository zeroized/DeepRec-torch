import pandas as pd
from feature.feature_meta import FeatureMeta
from preprocess.feat_engineering import preprocess_features


def load_and_preprocess(n_samples=10000, binarize_label=True):
    raw_ratings = pd.read_csv('G:\\dataset\\ml-latest-small\\ratings.csv')
    if n_samples > 0:
        raw_ratings = raw_ratings.sample(n=n_samples)
        raw_ratings.reset_index(drop=True, inplace=True)
    X = raw_ratings

    feature_meta = FeatureMeta()
    feature_meta.add_categorical_feat('userId')
    feature_meta.add_categorical_feat('movieId')

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
