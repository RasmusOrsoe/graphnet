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
        device=None,
    ):
        """DynEdge model.

        Args:
            nb_inputs (int): Number of input features.
            nb_outputs (int): Number of output features.
            layer_size_scale (int, optional): Integer that scales the size of
                hidden layers. Defaults to 4.
        """

        # pooling method
        self._coarsening = LONECoarsening(reduce="min")

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
                input_size=nb_inputs,
                hidden_size=hidden_size,
                features_subset=features_subset,
                device=device,
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
                    device=device,
                )
            )

        # Post-processing operations
        self.nn1 = torch.nn.Linear(nb_inputs + hidden_size * n_conv, l1)
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
        return data.x
