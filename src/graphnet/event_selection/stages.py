from abc import ABC, abstractmethod
import sqlite3
import pandas as pd
from typing import Any, List, Optional, Tuple, Union
import torch
import dill
from pytorch_lightning import Trainer

from graphnet.data.dataset import ColumnMissingException
from graphnet.utilities.logging import LoggerMixin
from graphnet.training.utils import get_predictions, make_dataloader
from graphnet.training.callbacks import ProgressBar

torch.multiprocessing.set_sharing_strategy("file_system")


class Stage(ABC, LoggerMixin):
    """Abstract Stage Class"""

    @property
    def name(self):
        return self.name

    @abstractmethod
    def __call__(self, df: pd.DataFrame):
        """Application Specific Method that runs the event selection."""


class MLInferenceStage(Stage):
    def __init__(self):
        assert self._config
        self._check_config(self._config)

    @property
    def _config(self):
        return self._config

    def __call__(self, database: str, selection: List[int]) -> pd.DataFrame:
        return self._inference(database, selection)

    def _load_model(self, path):
        return torch.load(path, map_location="cpu", pickle_module=dill)

    def _check_config(self, inference_config: dict) -> bool:
        required_fields = [
            "pickled_model_path",
            "pulse_map",
            "batch_size",
            "target",
            "features",
            "num_workers",
        ]
        for model_name in inference_config.keys():
            for required_field in required_fields:
                assert (
                    required_field in inference_config[model_name].keys()
                ), f"Inference config for {model_name} must contain {required_field}"
                try:
                    torch.load(
                        inference_config[model_name]["pickled_model_path"],
                        map_location="cpu",
                        pickle_module=dill,
                    )
                except Exception as e:
                    raise e
        return True

    def _merge_dataframes(
        main_df: pd.DataFrame, additional_df: pd.DataFrame
    ) -> pd.DataFrame:
        main_df = main_df.sort_values("event_no").reset_index(drop=True)
        additional_df = additional_df.sort_values("event_no").reset_index(
            drop=True
        )
        assert len(main_df) == len(
            additional_df
        ), "Dataframe1 does not contain the same amount of rows as Dataframe2"
        for column in additional_df.columns:
            if column not in main_df.columns:
                main_df[column] = additional_df[column]
        return main_df

    def _inference(
        self, database: str, selection: List[int], index_column: str
    ) -> pd.DataFrame:

        is_first = True
        for model_name in self._config.keys():
            trainer = Trainer(
                accelerator=self._config[model_name]["accelerator"],
                devices=self._config[model_name]["devices"],
                max_epochs=1,
                callbacks=[ProgressBar()],
                log_every_n_steps=1,
            )

            dataloader = make_dataloader(
                db=database,
                selection=selection,
                pulsemaps=self._config[model_name]["pulse_map"],
                features=self._config[model_name]["features"],
                truth=None,
                batch_size=self._config[model_name]["batch_size"],
                num_workers=self._config[model_name]["num_workers"],
                shuffle=False,
            )

            results = get_predictions(
                trainer=trainer,
                model=self._load_model(
                    self._config[model_name]["pickled_model_path"]
                ),
                dataloader=dataloader,
                prediction_columns=self._config["prediction_columns"],
                additional_attributes=[index_column],
            )
            if is_first:
                df = results
                is_first = False
            else:
                df = self._merge_dataframes(df, results)
        return df
