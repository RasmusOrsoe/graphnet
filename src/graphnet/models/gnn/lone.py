"""Implementation of the LONE super resolution GNN model architecture.

[Description of what this architecture does.]

Author: Rasmus Oersoe
Email: ###@###.###
"""
from typing import Callable, Optional, Sequence, Tuple, Union
import torch
import pandas as pd
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.nn import EdgeConv
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_sum
from graphnet.components.layers import DynEdgeConv, LONEConv
from graphnet.models.coarsening import DOMCoarsening

from graphnet.models.gnn.gnn import GNN
from graphnet.models.utils import calculate_xyzt_homophily


class LONE(GNN):
    def __init__(
        self,
        nb_inputs: int = 4,
        hidden_size: int = 256,
        n_conv: int = 1,
        max_k: int = 50,
        layer_size_scale: int = 4,
        pmt_idx: int = 0,
        string_idx: int = 1,
        time_idx: int = 2,
        custom_f2k=None,
    ):
        """DynEdge model.

        Args:
            nb_inputs (int): Number of input features.
            nb_outputs (int): Number of output features.
            layer_size_scale (int, optional): Integer that scales the size of
                hidden layers. Defaults to 4.
        """

        self._template_features = ["x", "y", "z", "time"]

        # custom f2k file from prometheus
        self._setup_geometry(custom_f2k)

        # member variables
        self._pmt_idx = pmt_idx
        self._string_idx = string_idx
        self._time_idx = time_idx

        # Architecture configuration
        c = layer_size_scale
        l1, l2, l3, l4, l5, l6 = (
            nb_inputs,
            c * 16 * 2,
            c * 32 * 2,
            c * 42 * 2,
            c * 32 * 2,
            c * 16 * 2,
        )
        l2 = l2
        # Base class constructor
        super().__init__(nb_inputs, l6)

        # Graph convolutional operations
        features_subset = slice(0, 3)
        self._convolution_layers = []
        # First convolution layer that upscales input to hidden_size
        self._convolution_layers.append(
            LONEConv(
                aggr="add",
                max_k=max_k,
                input_size=nb_inputs,
                hidden_size=hidden_size,
                features_subset=features_subset,
            )
        )
        # Number of hidden convolutional layers
        for i in range(n_conv - 1):
            self._convolution_layers.append(
                LONEConv(
                    aggr="add",
                    max_k=max_k,
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    features_subset=features_subset,
                )
            )

        # Post-processing operations
        self.nn1 = torch.nn.Linear(l3 * 4 + l1, l4)
        self.nn2 = torch.nn.Linear(l4, l5)
        self.nn3 = torch.nn.Linear(4 * l5 + 5, l6)
        self.lrelu = torch.nn.LeakyReLU()

    def forward(self, data: Data) -> Tensor:
        """Model forward pass.

        Args:
            data (Data): Graph of input features.

        Returns:
            Tensor: Model output.
        """
        # Query Table
        data = self._get_template(data)

        # Convenience variables
        x = data.x

        # Calculate homophily (scalar variables)
        # h_x, h_y, h_z, h_t = calculate_xyzt_homophily(x, edge_index, batch)

        # Convolutions
        for i in range(len(self._convolution_layers)):
            data = self._convolution_layers[i](data)
            x = torch.cat((x, data.x), dim=1)

        # Post-processing
        x = self.nn1(x)
        x = self.lrelu(x)
        x = self.nn2(x)

        # Read-out
        x = self.lrelu(x)
        x = self.nn3(x)
        x = self.lrelu(x)
        return x

    def _get_template(self, data):
        data_list = data.to_list()
        for graph in data_list:
            template = self._detector_template.clone()
            for pulse in range(len(graph.x)):
                template_idx = self._query_lookup_table(
                    graph.x[pulse, self._string_idx].item(),
                    graph.x[pulse, self._string_idx].item(),
                )
                template[
                    template_idx, self._lookup_features.index("time")
                ] = graph.x[pulse, self._time_idx]
            graph.x = template
            graph["active_doms"] = (
                template[:, self._lookup_features.index("time")] != 0
            ).long()
        return Batch.from_data_list(data_list)

    def _query_lookup_table(self, string_idx, pmt_idx):
        return self._lookup_table[string_idx][pmt_idx]

    def _make_lookup_table(self, geometry_table):
        """Creates a lookup table that matches pmt and string indices with row index in geometry table

        Args:
            geometry_table (pandas.DataFrame): the geometry table
        """
        table = {}
        for table_idx in range(len(geometry_table)):
            string_idx = geometry_table["string_idx"][table_idx]
            pmt_idx = geometry_table["pmt_idx"][table_idx]
            table[string_idx] = {}
            table[string_idx][pmt_idx] = table_idx
        self._lookup_table = table
        return

    def _make_detector_template(self, geometry_table):
        """Creates a template of the detector geometry for slicing later.

        Args:
            geometry_table (pandas.DataFrame): the geometry table
        """
        template = geometry_table.loc[:, ["x", "y", "z"]]
        template["time"] = 0
        self._detector_template = torch.tensor(template.values)
        return

    def _setup_geometry(self, custom_f2k):
        geometry_table = pd.read_csv(
            custom_f2k, sep="\t", lineterminator="\n", header=None
        )
        geometry_table.columns = [
            "hash_1",
            "hash_1",
            "x",
            "y",
            "z",
            "string_idx",
            "pmt_idx",
        ]
        self._make_lookup_table(geometry_table)
        self._make_detector_template(geometry_table)
        return
