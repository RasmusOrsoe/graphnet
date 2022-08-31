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
from graphnet.models.coarsening import LONECoarsening

from graphnet.models.gnn.gnn import GNN
from graphnet.models.utils import calculate_xyzt_homophily


class LONE(GNN):
    def __init__(
        self,
        nb_inputs: int = 4,
        hidden_size: int = 12,
        n_conv: int = 1,
        max_k: int = 50,
        layer_size_scale: int = 4,
        pmt_idx: int = 0,
        string_idx: int = 1,
        time_idx: int = 2,
        custom_f2k=None,
        device="cpu",
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

        # pooling method
        self._coarsening = LONECoarsening(reduce="min")

        # member variables
        self._pmt_idx = pmt_idx
        self._string_idx = string_idx
        self._time_idx = time_idx
        if device != "cpu":
            self._device = "cuda:%s" % device

        # Architecture configuration
        l1 = layer_size_scale * 12
        l2 = layer_size_scale * l1 * 4
        l3 = layer_size_scale * l1 * 2
        # Base class constructor
        super().__init__(nb_inputs, l3)

        # Graph convolutional operations
        features_subset = slice(0, 3)
        self._convolution_layers = []
        # First convolution layer that upscales input to hidden_size
        self._convolution_layers.append(
            LONEConv(
                aggr="add",
                max_k=max_k,
                input_size=nb_inputs + 1,
                hidden_size=hidden_size,
                features_subset=features_subset,
                device=self._device,
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
                    device=self._device,
                )
            )

        # Post-processing operations
        self.nn1 = torch.nn.Linear(nb_inputs + 1 + hidden_size * n_conv, l1)
        self.nn2 = torch.nn.Linear(l1, l2)
        self.nn3 = torch.nn.Linear(l2, l3)
        self.lrelu = torch.nn.LeakyReLU()

    def forward(self, data: Data) -> Tensor:
        """Model forward pass.

        Args:
            data (Data): Graph of input features.

        Returns:
            Tensor: Model output.
        """
        # DOM-Pooling - keeps only one pulse pr. pmt
        data = self._coarsening(data)
        # Query Table - matches keys to DOMs, including empty DOMs
        data = self._get_template(data)

        # Convenience variables
        x = data.x

        # Calculate homophily (scalar variables)
        # h_x, h_y, h_z, h_t = calculate_xyzt_homophily(x, edge_index, batch)

        # Convolutions
        for i in range(len(self._convolution_layers)):
            data = self._convolution_layers[i](data)
            x = torch.cat((x, data.x), dim=1)
        x = x.to(self._device)
        # Post-processing
        x = self.nn1(x)
        x = self.lrelu(x)
        x = self.nn2(x)

        # Read-out
        x = self.lrelu(x)
        x = self.nn3(x)
        data.x = self.lrelu(x)
        data.x
        return data

    def _get_template(self, data):
        data_list = data.to_data_list()
        for graph in data_list:
            template = self._detector_template.clone()
            for pulse in range(len(graph.x)):
                # because pmt_idx in pulsemap != pmt_idx in f2k-file
                # template_idx = self._query_lookup_table(
                #    graph.x[pulse, self._string_idx].item(),
                #    graph.x[pulse, self._pmt_idx].item(),
                # )
                template[
                    int(graph.x[pulse, self._pmt_idx].item()),
                    self._template_features.index("time"),
                ] = graph.x[pulse, self._time_idx]
            graph.x = template
            graph["active_doms"] = (
                (template[:, self._template_features.index("time")] != 0)
                .long()
                .reshape(-1, 1)
            )
        return Batch.from_data_list(data_list)

    def _query_lookup_table(self, string_idx, pmt_idx):
        return self._lookup_table[int(string_idx)][int(pmt_idx)]

    def _make_lookup_table(self, geometry_table):
        """Creates a lookup table that matches pmt and string indices with row index in geometry table

        Args:
            geometry_table (pandas.DataFrame): the geometry table
        """
        table = {}
        for table_idx in range(len(geometry_table)):
            string_idx = int(geometry_table["string_idx"][table_idx])
            pmt_idx = int(geometry_table["pmt_idx"][table_idx])
            try:
                table[string_idx]
            except KeyError:
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
        self._detector_template = torch.tensor(
            template.values, dtype=torch.float
        )
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
