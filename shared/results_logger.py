import csv
import os
from typing import Dict


class ExperimentLogger:
    def __init__(self, csv_path: str, fieldnames = None):
        self.csv_path = csv_path
        if fieldnames is None:
            self.fieldnames = [
                "experiment_id",
                "dataset_root",
                "matcher",
                # "matcher_config",
                "num_pairs",
                "match_points_total",
                "model_time",
                "pba_time",
                "mean_error",
                "median_error",
                "rmse",
                "max_error",
                "lt_1pix_percent",
                "lt_3pix_percent",
                "lt_5pix_percent",
            ]
        else:
            self.fieldnames = fieldnames

    def append(self, row: Dict):
        dir_name = os.path.dirname(self.csv_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        file_exists = os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in self.fieldnames})
