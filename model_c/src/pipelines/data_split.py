import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Text

from model_a.src.utils.config import load_config


def data_split(config_path: Text) -> None:
    """Split dataset into train/test.
    Args:
        config_path {Text}: path to config
    """
    config = load_config(config_path)

    features_path  = os.path.join(config.base.dir_data_processed, config.featurize.features_path)
    dataset = pd.read_csv(features_path)
    train_dataset, test_dataset = train_test_split(
        dataset,
        test_size=config.data_split.test_size,
        random_state=config.base.random_state
    )

    train_csv_path  = os.path.join(config.base.dir_data_processed, config.data_split.train_path)
    print(f'Save features to {train_csv_path}')
    train_dataset.to_csv(train_csv_path, index=False)
    
    test_csv_path  = os.path.join(config.base.dir_data_processed, config.data_split.test_path)
    print(f'Save features to {test_csv_path}')
    test_dataset.to_csv(test_csv_path, index=False)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data_split(config_path=args.config)
