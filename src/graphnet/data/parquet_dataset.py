from typing import List, Optional, Union
import pandas as pd
import numpy as np
import awkward as ak
import torch
from torch_geometric.data import Data
import time


class ParquetDataset(torch.utils.data.Dataset):
    """Pytorch dataset for reading from SQLite."""

    def __init__(
        self,
        path: str,
        pulsemaps: Union[str, List[str]],
        features: List[str],
        truth: List[str],
        node_truth: Optional[List[str]] = None,
        index_column: str = "event_no",
        truth_table: str = "truth",
        node_truth_table: Optional[str] = None,
        string_selection: Optional[List[int]] = None,
        string_index_column: str = "string_idx",
        selection: Optional[List[int]] = None,
        dtype: torch.dtype = torch.float32,
        pid_column: str = "pid",
        interaction_type_column: str = "interaction_type",
    ):

        # Check(s)
        if isinstance(path, list):
            print("multiple folders not supported")
            assert isinstance(1, str)
        assert isinstance(path, str)

        if isinstance(pulsemaps, str):
            pulsemaps = pulsemaps

        assert isinstance(features, (list, tuple))
        assert isinstance(truth, (list, tuple))
        if node_truth is not None:
            assert isinstance(node_truth, (list, tuple))

        self._interaction_type_column = interaction_type_column
        self._string_selection = string_selection
        self._string_index_column = string_index_column
        self._path = path
        self._pid_column = pid_column
        self._pulsemaps = pulsemaps
        self._features = features
        self._truth = truth
        self._index_column = index_column
        self._truth_table = truth_table
        self._node_truth = node_truth
        self._node_truth_table = node_truth_table
        self._dtype = dtype
        self._parquet_hook = ak.from_parquet(path)

        if selection is None:
            self._indices = self._get_all_indices()
        else:
            self._indices = selection

    def __len__(self):
        return len(self._indices)

    def _query_parquet(
        self,
        columns: Union[List, str],
        table: str,
        index: int,
        selection: Optional[str] = None,
    ):
        event_no = selection[index]
        query = self._parquet_hook[table][event_no][columns].to_list()
        return np.array([query[column] for column in columns]).T

    def __getitem__(self, i):
        features, truth, node_truth = self._get_event_data(i)
        if self._string_selection is not None:
            features = self._apply_selection_to_pulsemap(
                features, self._string_selection, self._string_index_column
            )
        graph = self._create_graph(features, truth, node_truth)
        return graph

    def _get_all_indices(self):
        return ak.to_numpy(self._parquet_hook[self._index_column]).tolist()

    def _apply_selection_to_pulsemap(self, pulsemap, selection, column_name):
        mask = []
        for item in selection:
            mask.extend(
                np.where(
                    pulsemap[:, self._features.index(column_name)] == item
                ).tolist()
            )
        return pulsemap[mask, :]

    def _get_event_data(self, i):
        """Query SQLite database for event feature and truth information.

        Args:
            i (int): Sequentially numbered index (i.e. in [0,len(self))) of the
                event to query.

        Returns:
            list: List of tuples, containing event features.
            list: List of tuples, containing truth information.
            list: List of tuples, containing node-level truth information.
        """
        features = self._query_parquet(
            self._features, self._pulsemaps, i, self._selection
        )
        truth = self._query_parquet(self._truth, self._truth_table, i)
        if self._node_truth is not None:
            node_truth = self._query_parquet(
                self._node_truth_column, self._node_truth_table, i
            )
        node_truth = None
        return features, truth, node_truth

    def _create_graph(self, features, truth, node_truth=None):
        """Create Pytorch Data (i.e.graph) object.

        No preprocessing is performed at this stage, just as no node adjancency
        is imposed. This means that the `edge_attr` and `edge_weight` attributes
        are not set.

        Args:
            features (list): List of tuples, containing event features.
            truth (list): List of tuples, containing truth information.
            node_truth (list): List of tuples, containing node-level truth.

        Returns:
            torch.Data: Graph object.
        """
        # Convert nested list to simple dict
        truth_dict = {key: truth[ix] for ix, key in enumerate(self._truth)}

        # assert len(truth) == len(self._truth)

        # Convert nested list to simple dict
        if node_truth is not None:
            node_truth_array = np.asarray(node_truth)
            node_truth_dict = {
                key: node_truth_array[:, ix]
                for ix, key in enumerate(self._node_truth)
            }

        # Unpack common variables
        abs_pid = abs(truth_dict[self._pid_column])

        labels_dict = {
            "event_no": truth_dict[self._index_column],
            "muon": int(abs_pid == 13),
            # "muon_stopped": int(truth_dict.get("stopped_muon") == 1),
            # "noise": int((abs_pid == 1) & (sim_type != "data")),
            "neutrino": int((abs_pid != 13) & (abs_pid != 1)),
            "v_e": int(abs_pid == 12),
            "v_u": int(abs_pid == 14),
            "v_t": int(abs_pid == 16),
            "track": int(
                (abs_pid == 14)
                & (truth_dict[self._interaction_type_column] == 1)
            ),
        }

        # Construct graph data object
        x = torch.tensor(features, dtype=self._dtype)
        n_pulses = torch.tensor(len(x), dtype=torch.int32)
        graph = Data(x=x, edge_index=None)
        graph.n_pulses = n_pulses
        graph.features = self._features

        # Write attributes, either target labels, truth info or original features.
        add_these_to_graph = [
            labels_dict,
            truth_dict,
        ]  # [labels_dict, truth_dict]
        if node_truth is not None:
            add_these_to_graph.append(node_truth_dict)
        for write_dict in add_these_to_graph:
            for key, value in write_dict.items():
                try:
                    graph[key] = torch.tensor(value)
                except TypeError:
                    # Cannot convert `value` to Tensor due to its data type, e.g. `str`.
                    pass
        for ix, feature in enumerate(graph.features):
            graph[feature] = graph.x[:, ix].detach()
        return graph
