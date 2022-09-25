"""Module defining the base `Dataset` class used in GraphNeT."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from graphnet.utilities.logging import LoggerMixin


class ColumnMissingException(Exception):
    """Exception to indicate a missing column in a dataset."""


class Dataset(ABC, torch.utils.data.Dataset, LoggerMixin):
    """Base Dataset class for reading from any intermediate file format."""

    def __init__(
        self,
        path: Union[str, List[str]],
        pulsemaps: Union[str, List[str]],
        features: List[str],
        truth: List[str],
        *,
        node_truth: Optional[List[str]] = None,
        index_column: str = "event_no",
        truth_table: str = "truth",
        node_truth_table: Optional[str] = None,
        string_selection: Optional[List[int]] = None,
        sensor_selection: Optional[List[int]] = None,
        selection: Optional[List[int]] = None,
        dtype: torch.dtype = torch.float32,
        loss_weight_table: Optional[str] = None,
        loss_weight_column: Optional[str] = None,
        loss_weight_default_value: Optional[float] = None,
        pmt_idx_column: str = "sensor_id",
        string_idx_column: str = "sensor_string_id",
        geometry_table: Optional[str] = None,
        include_inactive_sensors: bool = False,
        pid_column: str = "pid",
        interaction_type_column: str = "interaction_type",
    ):
        # Check(s)
        if isinstance(pulsemaps, str):
            pulsemaps = [pulsemaps]

        assert isinstance(features, (list, tuple))
        assert isinstance(truth, (list, tuple))

        # Member variable(s)
        self._path = path
        self._selection = None
        self._pulsemaps = pulsemaps
        self._features = [index_column] + features
        self._truth = [index_column] + truth
        self._index_column = index_column
        self._truth_table = truth_table
        self._loss_weight_default_value = loss_weight_default_value
        self._pmt_idx_column = pmt_idx_column
        self._string_idx_column = string_idx_column
        self._geometry_file = geometry_table
        self._include_inactive_sensors = include_inactive_sensors
        self._interaction_type_column = interaction_type_column
        self._pid_column = pid_column
        if geometry_table is not None:
            if self._include_inactive_sensors:
                self._detector_template = self._make_detector_template(
                    geometry_table
                )

        if node_truth is not None:
            assert isinstance(node_truth_table, str)
            if isinstance(node_truth, str):
                node_truth = [node_truth]

        self._node_truth = node_truth
        self._node_truth_table = node_truth_table

        if string_selection is not None:
            self.logger.warning(
                (
                    "String selection detected.\n",
                    f"Accepted strings: {string_selection}\n",
                    "All other strings are ignored!",
                )
            )
            if isinstance(string_selection, int):
                string_selection = [string_selection]

        self._string_selection = string_selection

        if sensor_selection is not None:
            self.logger.warning(
                (
                    "Sensor selection detected.\n",
                    f"Accepted sensors: {sensor_selection}\n",
                    "All other sensors are ignored!",
                )
            )
            if isinstance(sensor_selection, int):
                sensor_selection = [sensor_selection]

        self._sensor_selection = sensor_selection

        self._selection = None
        if self._string_selection:
            self._selection = f"{self._string_idx_column} in {str(tuple(self._string_selection))}"
        if self._sensor_selection:
            if self._selection is None:
                self._selection = f"{self._pmt_idx_column} in {str(tuple(self._sensor_selection))}"
            else:
                self._selection = (
                    self._selection
                    + f" AND {self._pmt_idx_column} in {str(tuple(self._sensor_selection))}"
                )

        self._loss_weight_column = loss_weight_column
        self._loss_weight_table = loss_weight_table
        if (self._loss_weight_table is None) and (
            self._loss_weight_column is not None
        ):
            self.logger.warning("Error: no loss weight table specified")
            assert isinstance(self._loss_weight_table, str)
        if (self._loss_weight_table is not None) and (
            self._loss_weight_column is None
        ):
            self.logger.warning("Error: no loss weight column specified")
            assert isinstance(self._loss_weight_column, str)

        self._dtype = dtype

        # Implementation-specific initialisation.
        self._init()

        # Set unique indices
        if selection is None:
            self._indices = self._get_all_indices()
        else:
            self._indices = selection

        # Purely internal member variables
        self._missing_variables = {}
        self._remove_missing_columns()

        # Implementation-specific post-init code.
        self._post_init()

    # Abstract method(s)
    @abstractmethod
    def _init(self):
        """Set internal representation needed to read data from input file."""

    def _post_init(self):
        """Implemenation-specific code to be run after the main constructor."""

    @abstractmethod
    def _get_all_indices(self) -> List[int]:
        """Return a list of all available values in `self._index_column`."""

    @abstractmethod
    def _query_table(
        self,
        table: str,
        columns: Union[List[str], str],
        index: int,
        selection: Optional[str] = None,
    ) -> List[Tuple[Any]]:
        """Query a table at a specific index, optionally with some selection.

        Args:
            table (str): Table to be queried.
            columns (List[str]): Columns to read out.
            index (int): Sequentially numbered index (i.e. in [0,len(self))) of
                the event to query. This _may_ differ from the indexation used
                in `self._indices`.
            selection (Optional[str], optional): Selection to be imposed before
                reading out data. Defaults to None.

        Returns:
            List[Tuple[Any]]: Returns a list of tuples containing the values in
                `columns`. If the `table` contains only scalar data for
                `columns`, a list of length 1 is returned

        Raises:
            ColumnMissingException: If one or more element in `columns` is not
                present in `table`.
        """

    # Public method(s)
    def __len__(self):
        return len(self._indices)

    def __getitem__(self, index: int) -> Data:
        if not (0 <= index < len(self)):
            raise IndexError(
                f"Index {index} not in range [0, {len(self) - 1}]"
            )
        features, truth, node_truth, loss_weight = self._query(index)
        graph = self._create_graph(features, truth, node_truth, loss_weight)
        if self._include_inactive_sensors:
            assert (
                self._detector_template is not None
            ), "Geometry file must be specified if inactive sensors are to be included"
            # graph = self._add_inactive_sensors(graph)
            # graph = self._add_active_sensor_labels(graph)
        return graph

    # Internal method(s)
    def _remove_missing_columns(self):
        """Remove columns that are not present in the input file.

        Columns are removed from `self._features` and `self._truth`.
        """
        # Find missing features
        missing_features = set(self._features)
        for pulsemap in self._pulsemaps:
            missing = self._check_missing_columns(self._features, pulsemap)
            missing_features = missing_features.intersection(missing)

        missing_features = list(missing_features)

        # Find missing truth variables
        missing_truth_variables = self._check_missing_columns(
            self._truth, self._truth_table
        )

        # Remove missing features
        if missing_features:
            self.logger.warning(
                "Removing the following (missing) features: "
                + ", ".join(missing_features)
            )
            for missing_feature in missing_features:
                self._features.remove(missing_feature)

        # Remove missing truth variables
        if missing_truth_variables:
            self.logger.warning(
                (
                    "Removing the following (missing) truth variables: "
                    + ", ".join(missing_truth_variables)
                )
            )
            for missing_truth_variable in missing_truth_variables:
                self._truth.remove(missing_truth_variable)

    def _check_missing_columns(
        self,
        columns: List[str],
        table: str,
    ) -> List[str]:
        """Return a list missing columns in `table`."""
        for column in columns:
            try:
                self._query_table(table, [column], 0)
            except ColumnMissingException:
                if table not in self._missing_variables:
                    self._missing_variables[table] = []
                self._missing_variables[table].append(column)

        return self._missing_variables.get(table, [])

    def _query(
        self, index: int
    ) -> Tuple[List[Tuple], List[Tuple], List[Tuple]]:
        """Query file for event features and truth information

        Args:
            index (int): Sequentially numbered index (i.e. in [0,len(self))) of
                the event to query. This _may_ differ from the indexation used
                in `self._indices`.

        Returns:
            List[Tuple]: Pulse-level event features.
            List[Tuple]: Event-level truth information. List has length 1.
            List[Tuple]: Pulse-level truth information.
        """

        features = []
        for pulsemap in self._pulsemaps:
            features_pulsemap = self._query_table(
                pulsemap, self._features, index, self._selection
            )
            features.extend(features_pulsemap)

        truth = self._query_table(self._truth_table, self._truth, index)
        if self._node_truth:
            node_truth = self._query_table(
                self._node_truth_table,
                self._node_truth,
                index,
                self._selection,
            )
        else:
            node_truth = None

        loss_weight = None  # Default
        if self._loss_weight_column is not None:
            if self._loss_weight_table is not None:
                loss_weight = self._query_table(
                    self._loss_weight_table, self._loss_weight_column, index
                )
        return features, truth, node_truth, loss_weight

    def _create_graph(
        self,
        features: List[Tuple[Any]],
        truth: List[Tuple[Any]],
        node_truth: Optional[List[Tuple[Any]]] = None,
        loss_weight: Optional[float] = None,
    ) -> Data:
        """Create Pytorch Data (i.e.graph) object.

        No preprocessing is performed at this stage, just as no node adjancency
        is imposed. This means that the `edge_attr` and `edge_weight`
        attributes are not set.

        Args:
            features (list): List of tuples, containing event features.
            truth (list): List of tuples, containing truth information.
            node_truth (list): List of tuples, containing node-level truth.
            loss_weight (float): A weight associated with the event for weighing the loss.

        Returns:
            torch.Data: Graph object.
        """
        # Convert nested list to simple dict
        truth_dict = {
            key: truth[0][index] for index, key in enumerate(self._truth)
        }
        assert len(truth) == 1

        # Define custom labels
        labels_dict = self._get_labels(truth_dict)

        # Convert nested list to simple dict
        if node_truth is not None:
            node_truth_array = np.asarray(node_truth)
            node_truth_dict = {
                key: node_truth_array[:, index]
                for index, key in enumerate(self._node_truth)
            }

        # Catch cases with no reconstructed pulses
        if len(features):
            data = np.asarray(features)[:, 1:]
        else:
            data = np.array([]).reshape((0, len(self._features) - 1))

        # Construct graph data object
        x = torch.tensor(data, dtype=self._dtype)  # pylint: disable=C0103
        n_pulses = torch.tensor(len(x), dtype=torch.int32)
        if self._include_inactive_sensors:
            assert (
                self._detector_template is not None
            ), "Geometry file must be specified if inactive sensors are to be included"
            x = self._add_inactive_sensors(x)
            graph = Data(x=x, edge_index=None)
            graph = self._add_active_sensor_labels(graph)
            graph["dom_x"] = x[:, 0]
            graph["dom_y"] = x[:, 1]
            graph["dom_z"] = x[:, 2]
            graph["full_grid_time"] = x[:, 3]
            graph.features = ["dom_x", "dom_y", "dom_z", "full_grid_time"]
        else:
            graph = Data(x=x, edge_index=None)
            graph.features = self._features[1:]
        graph.n_pulses = n_pulses

        # Add loss weight to graph.
        if loss_weight is not None and self._loss_weight_column is not None:
            # No loss weight was retrieved, i.e., it is missing for the current event
            if len(loss_weight) == 0:
                if self._loss_weight_default_value is None:
                    raise ValueError(
                        "At least one event is missing an entry in "
                        f"{self._loss_weight_column} "
                        "but loss_weight_default_value is None."
                    )
                graph[self._loss_weight_column] = torch.tensor(
                    self._loss_weight_default_value, dtype=self._dtype
                ).reshape(-1, 1)
            else:
                graph[self._loss_weight_column] = torch.tensor(
                    loss_weight, dtype=self._dtype
                ).reshape(-1, 1)

        # Write attributes, either target labels, truth info or original
        # features.
        add_these_to_graph = [labels_dict, truth_dict]
        if node_truth is not None:
            add_these_to_graph.append(node_truth_dict)
        for write_dict in add_these_to_graph:
            for key, value in write_dict.items():
                try:
                    graph[key] = torch.tensor(value)
                except TypeError:
                    # Cannot convert `value` to Tensor due to its data type,
                    # e.g. `str`.
                    self.logger.debug(
                        (
                            f"Could not assign `{key}` with type "
                            f"'{type(value).__name__}' as attribute to graph."
                        )
                    )

        # Additionally add original features as (static) attributes
        for index, feature in enumerate(graph.features):
            graph[feature] = graph.x[:, index].detach()
        # print(graph['event_no'])
        return graph

    def _get_labels(self, truth_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Return dictionary of  labels, to be added as graph attributes."""
        abs_pid = abs(truth_dict[self._pid_column])

        labels_dict = {
            self._index_column: truth_dict[self._index_column],
            "muon": int(abs_pid == 13),
            "neutrino": int((abs_pid != 13) & (abs_pid != 1)),
            "v_e": int(abs_pid == 12),
            "v_u": int(abs_pid == 14),
            "v_t": int(abs_pid == 16),
            "track": int(
                (abs_pid == 14)
                & (truth_dict[self._interaction_type_column] == 1)
            ),
        }
        return labels_dict

    def _add_inactive_sensors(self, x: torch.tensor):
        template = self._detector_template.clone()
        same_pmt_pulses = None
        for pulse in range(len(x)):
            if (
                template[
                    int(
                        x[
                            pulse, self._features.index(self._pmt_idx_column)
                        ].item()
                    ),
                    3,
                ]
                == 0
            ):
                template[
                    int(
                        x[
                            pulse, self._features.index(self._pmt_idx_column)
                        ].item()
                    ),
                    3,
                ] = x[pulse, self._features.index("t") - 1]
            else:
                if same_pmt_pulses is None:
                    same_pmt_pulses = template[
                        int(
                            x[
                                pulse,
                                self._features.index(self._pmt_idx_column),
                            ].item()
                        ),
                        :,
                    ].reshape(1, -1)
                    same_pmt_pulses[:, 3] = x[
                        pulse, self._features.index("t") - 1
                    ]
                else:
                    append_this = template[
                        int(
                            x[
                                pulse,
                                self._features.index(self._pmt_idx_column),
                            ].item()
                        ),
                        :,
                    ].reshape(1, -1)
                    append_this[:, 3] = x[pulse, self._features.index("t") - 1]
                    same_pmt_pulses = torch.cat(
                        [same_pmt_pulses, append_this], dim=0
                    )
        if same_pmt_pulses is not None:
            pass  # template = torch.cat([template, same_pmt_pulses], dim=0)

        return template

    def _add_active_sensor_labels(self, graph: Data):
        graph["active_doms"] = (graph.x[:, 3] != 0).long()  # .reshape(-1, 1)
        return graph

    def _make_detector_template(self, geometry_table):
        """Creates a template of the detector geometry for slicing later.
        Args:
            geometry_table (str): path the geometry table where each row is a sensor module. Must contain xyz positions of each module.
        """
        # geometry_table = pd.read_csv(geometry_table)
        template = geometry_table.loc[:, ["sensor_x", "sensor_y", "sensor_z"]]
        template["time"] = 0
        return torch.tensor(template.values, dtype=torch.float)
