import argparse
import joblib
import os
import pandas as pd
from typing import Text

from model_a.src.train.train import train
from model_a.src.utils.config import load_config


def train_model(config_path: Text) -> None:
    """Train model.
    Args:
        config_path {Text}: path to config
    """

    config = load_config(config_path)
    estimator_name = config.train.estimator_name
    param_grid = config.train.estimators[estimator_name].param_grid
    cv = config.train.cv
    target_column = config.featurize.target_column
    
    train_csv_path  = os.path.join(config.base.dir_data_processed, config.data_split.train_path)
    train_df = pd.read_csv(train_csv_path)

    model = train(
        df=train_df,
        target_column=target_column,
        estimator_name=estimator_name,
        param_grid=param_grid,
        cv=cv
    )
    print(model.best_score_)

    model_path = os.path.join(config.base.dir_models, config.train.model_name)
    print(f'Save model A to {model_path}')
    joblib.dump(model, model_path)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train_model(config_path=args.config)
