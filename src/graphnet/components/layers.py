from typing import Callable, Optional, Sequence, Tuple, Union
from torch.functional import Tensor
import torch
from torch_geometric.nn import EdgeConv
from torch.nn import Linear, Sequential, LeakyReLU, Softmax
from torch_geometric.nn.pool import knn_graph
from torch_geometric.data import Data, Batch
from torch_geometric.typing import Adj
from graphnet.models.utils import knn_graph_batch
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_sum
import numpy as np
from torch_geometric.nn.conv import MessagePassing


class LONEConv(MessagePassing):
    def __init__(
        self,
        aggr: str = "max",
        max_k: int = 50,
        input_size: int = 4,
        hidden_size: int = 256,
        features_subset: Optional[Sequence] = None,
        device=None,
        **kwargs,
    ):
        super(LONEConv, self).__init__()
        # Check(s)
        if features_subset is None:
            features_subset = slice(None)  # Use all features
        assert isinstance(features_subset, (list, slice))

        # Additional member variables
        self.features_subset = features_subset
        self.input_mlp = Sequential(
            Linear(input_size, hidden_size),
            LeakyReLU(),
            Linear(hidden_size, hidden_size),
            LeakyReLU(),
        ).to(device)

        self.scalar_mlp = Sequential(
            Linear(input_size * 4, hidden_size),
            LeakyReLU(),
            Linear(hidden_size, hidden_size),
            LeakyReLU(),
        ).to(device)

        self.k_predictor = Sequential(
            Linear(hidden_size, hidden_size),
            LeakyReLU(),
            Linear(hidden_size, max_k),
            Softmax(),
        ).to(device)

        self.EdgeConv = EdgeConv(
            nn=torch.nn.Sequential(
                torch.nn.Linear(input_size * 2, hidden_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.LeakyReLU(),
            ),
            aggr=aggr,
        ).to(device)

    def forward(self, data: Union[Data, Batch]) -> Tensor:
        # fork data flow to k predictor
        x = self.input_mlp(data.x)
        # Simple Node Aggregation
        a, _ = scatter_max(data.x, data.batch, dim=0)
        b, _ = scatter_min(data.x, data.batch, dim=0)
        c = scatter_sum(data.x, data.batch, dim=0)
        d = scatter_mean(data.x, data.batch, dim=0)
        x = torch.cat((a, b, c, d), dim=1)
        # Pass aggregated data through MLP
        x = self.scalar_mlp(x)
        # predict neighbourhood size of each graph in batch based on forked, aggregated data
        k_list = np.argmax(
            self.k_predictor(x).detach().cpu().numpy(), axis=1
        ).tolist()
        # Compute adjacency on data.x given predicted k_list
        data = knn_graph_batch(data, k=k_list, columns=self.features_subset)
        # Standard EdgeConv forward pass on data.x
        data.x = self.EdgeConv(data.x, data.edge_index)
        return data


class DynEdgeConv(EdgeConv):
    def __init__(
        self,
        nn: Callable,
        aggr: str = "max",
        nb_neighbors: int = 8,
        features_subset: Optional[Sequence] = None,
        **kwargs,
    ):
        # Check(s)
        if features_subset is None:
            features_subset = slice(None)  # Use all features
        assert isinstance(features_subset, (list, slice))

        # Base class constructor
        super().__init__(nn=nn, aggr=aggr, **kwargs)

        # Additional member variables
        self.nb_neighbors = nb_neighbors
        self.features_subset = features_subset

    def forward(
        self, x: Tensor, edge_index: Adj, batch: Optional[Tensor] = None
    ) -> Tensor:
        # Standard EdgeConv forward pass
        x = super().forward(x, edge_index)

        # Recompute adjacency
        edge_index = knn_graph(
            x=x[:, self.features_subset],
            k=self.nb_neighbors,
            batch=batch,
        ).to(x.device)

        return x, edge_index
