import argparse
import joblib
import json
import os
import pandas as pd
from typing import Text

from model_a.src.data.dataset import get_target_names
from model_a.src.evaluate.evaluate import evaluate
from model_a.src.report.visualize import plot_confusion_matrix
from model_a.src.utils.config import load_config


def evaluate_model(config_path: Text) -> None:
    """Evaluate model.
    Args:
        config_path {Text}: path to config
    """

    config = load_config(config_path)
    model_path = os.path.join(config.base.dir_models, config.train.model_name)
    model = joblib.load(model_path)
    test_csv_path  = os.path.join(config.base.dir_data_processed, config.data_split.test_path)
    test_df = pd.read_csv(test_csv_path)
    

    report = evaluate(df=test_df,
                      target_column=config.featurize.target_column,
                      clf=model)
    classes = get_target_names()

    # save f1 metrics file
    metrics_path = os.path.join(config.base.dir_reports, config.evaluate.metrics_file)
    json.dump(
        obj={'f1_score': report['f1']},
        fp=open(metrics_path, 'w')
    )
    print(f'F1 metrics file saved to : {metrics_path}')

    # save confusion_matrix.png
    plt = plot_confusion_matrix(cm=report['cm'],
                                target_names=get_target_names(),
                                normalize=False)
    confusion_matrix_png_path = os.path.join(config.base.dir_reports, config.evaluate.confusion_matrix_png)
    plt.savefig(confusion_matrix_png_path)
    print(f'Confusion matrix saved to : {confusion_matrix_png_path}')

    # save confusion_matrix.json
    classes_path = os.path.join(config.base.dir_reports, config.evaluate.plots_file)
    mapping = {
        0: classes[0],
        1: classes[1],
        2: classes[2]
    }
    df = (pd.DataFrame({'actual': report['actual'],
                        'predicted': report['predicted']})
          .assign(actual=lambda x: x.actual.map(mapping))
          .assign(predicted=lambda x: x.predicted.map(mapping))
          )
    df.to_csv(classes_path, index=False)
    print(f'Classes actual/predicted saved to : {classes_path}')


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    evaluate_model(config_path=args.config)
