import argparse
import json
import os
from typing import Text

from src.utils.config import load_config


def build_metrics_report(config_path: Text) -> None:
    """Create new features.
    Args:
        config_path {Text}: path to config
    """

    config = load_config(config_path)
    metrics = {}
    
    for model in ['model_a', 'model_b', 'model_c']:
        metrics_path = os.path.join(config.base.dir_reports, config[model]['metrics_file'])

        with open(metrics_path, 'r') as fp :
            metrics_dict = json.load(fp)
        
        for metric_name in metrics_dict.keys():
            
            # create an empty dict for a new metric_name 
            metrics.setdefault(metric_name, {})
            
            # add the metric score for the model
            metrics[metric_name].update({model: metrics_dict.get(metric_name, None)})

    # Save metrics report
    metrics_path = os.path.join(config.base.dir_reports, config.build_metrics_report.metrics_file)
    json.dump(metrics, fp=open(metrics_path, 'w'))
    print(f'Metrics report saved to : {metrics_path}')
    

if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    build_metrics_report(config_path=args.config)
