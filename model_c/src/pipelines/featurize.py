import argparse
import joblib
import os
import pandas as pd
from typing import Text

from model_a.src.features.features import extract_features
from model_a.src.utils.config import load_config


def featurize(config_path: Text) -> None:
    """Create new features.
    Args:
        config_path {Text}: path to config
    """

    config = load_config(config_path)

    dataset_path = os.path.join(config.base.dir_data_raw, config.data_load.dataset_csv)
    dataset = pd.read_csv(dataset_path)
    featured_dataset = extract_features(dataset)
    y = featured_dataset.loc[:, config.featurize.target_column].values.astype('int32')
    
    # Get Model A predictions 
    model_a_path = os.path.join(config.base.dir_models, config.featurize.dependencies.model_a)
    model_a = joblib.load(model_a_path)
    X = featured_dataset.drop(config.featurize.target_column, axis=1).values.astype('float32')
    preds_a = model_a.predict(X)
    # print(preds_a)
    
    # Get Model B predictions 
    model_b_path = os.path.join(config.base.dir_models, config.featurize.dependencies.model_b)
    model_b = joblib.load(model_b_path)
    X = featured_dataset.drop(config.featurize.target_column, axis=1).values.astype('float32')
    preds_b = model_b.predict(X)
    # print(preds_b)
    
    featured_dataset['model_a'] = preds_a
    featured_dataset['model_b'] = preds_b
    features_path  = os.path.join(config.base.dir_data_processed, config.featurize.features_path)
    print(f'Save features to {features_path}')
    featured_dataset.to_csv(features_path, index=False)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    featurize(config_path=args.config)
