"""Class(es) for constructing training labels at runtime."""

from abc import ABC, abstractmethod
import torch
from torch_geometric.data import Data
from graphnet.utilities.logging import Logger
from graphnet.models.graphs import GraphDefinition

from typing import List, Optional, Union

import sqlite3
import numpy as np


class Label(ABC, Logger):
    """Base `Label` class for producing labels from single `Data` instance."""

    def __init__(self, key: str):
        """Construct `Label`.

        Args:
            key: The name of the field in `Data` where the label will be
                stored. That is, `graph[key] = label`.
        """
        self._key = key

        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

    @property
    def key(self) -> str:
        """Return value of `key`."""
        return self._key

    @abstractmethod
    def __call__(self, graph: Data) -> torch.tensor:
        """Label-specific implementation."""


class Direction(Label):
    """Class for producing particle direction/pointing label."""

    def __init__(
        self,
        key: str = "direction",
        azimuth_key: str = "azimuth",
        zenith_key: str = "zenith",
    ):
        """Construct `Direction`.

        Args:
            key: The name of the field in `Data` where the label will be
                stored. That is, `graph[key] = label`.
            azimuth_key: The name of the pre-existing key in `graph` that will
                be used to access the azimiuth angle, used when calculating
                the direction.
            zenith_key: The name of the pre-existing key in `graph` that will
                be used to access the zenith angle, used when calculating the
                direction.
        """
        self._azimuth_key = azimuth_key
        self._zenith_key = zenith_key

        # Base class constructor
        super().__init__(key=key)

    def __call__(self, graph: Data) -> torch.tensor:
        """Compute label for `graph`."""
        x = torch.cos(graph[self._azimuth_key]) * torch.sin(
            graph[self._zenith_key]
        ).reshape(-1, 1)
        y = torch.sin(graph[self._azimuth_key]) * torch.sin(
            graph[self._zenith_key]
        ).reshape(-1, 1)
        z = torch.cos(graph[self._zenith_key]).reshape(-1, 1)
        return torch.cat((x, y, z), dim=1)


class SQLiteSuperResolutionLabel(Label):
    """A SQLite-specific label maker for SISR."""

    def __init__(
        self,
        graph_definition: GraphDefinition,
        string_selection: List[int],
        string_column: str,
        pulsemap: str,
        index_column: str = "event_no",
        key: str = "SR",
        drop_columns: List[str] = None,
    ) -> None:
        """Initialize.

        Args:
            graph_definition: graphdef
            string_selection: string selection
            string_column: column name for string index
            pulsemap: name of pulsemap
            index_column: event index column name. Defaults to "event_no".
            key: the name of the label. Defaults to "SR".
            drop_columns: columns to drop in label. Defaults to Optional[List[str]].
        """
        assert graph_definition._add_inactive_sensors is True
        assert graph_definition._sort_by is not None
        self._graph_definition = graph_definition
        self._string_selection = string_selection
        self._pulsemap = pulsemap
        self._string_column = string_column
        self._index_column = index_column

        detector = self._graph_definition._detector

        sensor_id_column = detector.sensor_id_column
        geometry_table = detector.geometry_table

        lookup = graph_definition._geometry_table_lookup(
            node_features=geometry_table.to_numpy(),
            node_feature_names=graph_definition._node_feature_names,
        )
        mask = ~geometry_table.loc[lookup, sensor_id_column].isin(
            graph_definition._sensor_mask
        )
        target_sensor_id = geometry_table.loc[
            mask, sensor_id_column
        ].to_numpy()
        if drop_columns is not None:
            self._keep_columns = self._create_column_mask(drop_columns)
            targets = []
            for feature in self._keep_columns:
                targets.append(
                    self._graph_definition.output_feature_names[feature]
                )
        else:
            targets = self._graph_definition.output_feature_names
        all_targets = []
        for id in target_sensor_id:
            for target in targets:
                all_targets.append(target + f"_sensor_{id}")

        self._targets = all_targets
        # Base class constructor
        super().__init__(key=key)

    def _query(self, graph: Data) -> np.ndarray:
        database = graph["dataset_path"]
        assert database.endswith(".db")
        with sqlite3.connect(database) as con:
            query = f'select {", ".join(self._graph_definition._node_feature_names)} from {self._pulsemap} where {self._string_column} not in {str(tuple(self._string_selection))} and {self._index_column} == {graph[self._index_column]}'
            a = np.asarray(con.execute(query).fetchall())
            if len(a) == 0:
                query = f'select {", ".join(self._graph_definition._node_feature_names)} from {self._pulsemap} where {self._string_column} not in {str(tuple(self._string_selection))} limit 1'
                a = np.asarray(con.execute(query).fetchall())
                for feature in self._graph_definition._node_feature_names:
                    if feature not in self._graph_definition._detector.xyz:
                        a[
                            :,
                            self._graph_definition._node_feature_names.index(
                                feature
                            ),
                        ] = 0
        return a

    def _create_column_mask(
        self, drop_columns: Union[List[str], None]
    ) -> List[int]:
        drop_idx = []
        assert drop_columns is not None  # mypy
        for feature in drop_columns:
            drop_idx.append(
                self._graph_definition.output_feature_names.index(feature)
            )

        keep_these = []
        for idx in np.arange(len(self._graph_definition.output_feature_names)):
            if idx not in drop_idx:
                keep_these.append(idx)
        return keep_these

    def __call__(self, graph: Data) -> torch.tensor:
        """Compute label for `graph`."""
        raw_target_pulses = self._query(graph)
        processed_target_pulses = self._graph_definition(
            raw_target_pulses, self._graph_definition._node_feature_names
        ).x
        if self._keep_columns is not None:
            processed_target_pulses = processed_target_pulses[
                :, self._keep_columns
            ]
        return processed_target_pulses.reshape(1, -1).squeeze(1)
