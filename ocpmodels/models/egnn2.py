"""
E (n) Equivariant Graph Neural Networks

This code implements the paper
E (n) Equivariant Graph Neural Networks
by Victor Garcia Satorras, Emiel Hoogeboom, Max Welling
https://arxiv.org/abs/2102.09844
---
"""

import numpy as np
import torch
from collections import OrderedDict
from torch import nn
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing, global_mean_pool, InstanceNorm
from torch_geometric.nn.inits import glorot_orthogonal

from ocpmodels.common.registry import registry
from ocpmodels.datasets.embeddings import CONTINUOUS_EMBEDDINGS
from ocpmodels.models.base import BaseModel
from ocpmodels.common.utils import (
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)

try:
    import sympy as sym
except ImportError:
    sym = None

class Swish(nn.Module):
    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta*x)


class EGNN_Layer(MessagePassing):
    """E(N) equivariant message passing neural network layer.
    Parameters
    ----------
    in_features : int
        Dimensionality of input features.
    out_features : int
        Dimensionality of output features.
    dim : int
        The "N" in the E(N) equivariance.
    update_pos : bool
        Flag whether to update position.
    """
    def __init__(self, in_features, out_features, hidden_features, dim=3, update_pos=True, infer_edges=True):
        super(EGNN_Layer, self).__init__(node_dim=-2, aggr='mean')
        self.update_pos = update_pos
        self.infer_edges = infer_edges
        self.dim = dim

        self.message_net = nn.Sequential(nn.Linear(2*in_features + 1, hidden_features),
                                        Swish(),
                                        nn.BatchNorm1d(hidden_features),
                                        nn.Linear(hidden_features, out_features),
                                        Swish()
                                        )
        self.update_net = nn.Sequential(nn.Linear(in_features + hidden_features, hidden_features),
                                        Swish(),
                                        nn.Linear(hidden_features, out_features),
                                        Swish()
                                        )
        if self.update_pos:
            self.pos_net = nn.Sequential(nn.Linear(hidden_features, hidden_features),
                                        Swish(),
                                        nn.Linear(hidden_features, 1)
                                        )
        if self.infer_edges:
            self.inf_net = nn.Sequential(nn.Linear(hidden_features, 1),
                                         nn.Sigmoid()
                                         )

        self.norm = InstanceNorm(hidden_features)

    def forward(self, x, pos, edge_index, cell_offsets, batch):
        """ Propagate messages along edges """
        x, pos = self.propagate(edge_index, x=x, pos=pos, cell_offsets=cell_offsets)
        x = self.norm(x, batch)
        return x, pos

    def message(self, x_i, x_j, pos_i, pos_j, cell_offsets):
        """ Message according to eqs 3-4 in the paper """
        distance_vectors = (pos_i - pos_j)+cell_offsets
        distance = distance_vectors.pow(2).sum(-1, keepdims=True)
        message = self.message_net(torch.cat((x_i, x_j, distance), dim=-1))

        if self.infer_edges:
            message = self.inf_net(message)*message

        if self.update_pos:
            pos_message = distance_vectors * self.pos_net(message)
            # torch geometric does not support tuple outputs.
            message = torch.cat((pos_message, message), dim=-1)
        return message

    def update(self, message, x, pos):
        """ Update according to eq 6 in the paper """
        pos_message = message[:, :self.dim]
        message = message[:, self.dim:]
        if self.update_pos:
            pos += pos_message
        update = self.update_net(torch.cat((x, message), dim=-1))
        x += update
        return x, pos

    
@registry.register_model("egnn2")
class EGNN(torch.nn.Module):
    def __init__(
        self,
        num_atoms,  # not used
        bond_feat_dim,  # not used
        num_targets,  # not used
        in_features,
        out_features,
        hidden_features,
        hidden_layer=7,
        dim=3,
        update_pos=True,
        regress_forces=False,
        use_pbc=True,
        otf_graph=False
    ):

        super(EGNN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.hidden_layer = hidden_layer
        self.regress_forces = regress_forces
        self.otf_graph = otf_graph
        self.use_pbc = use_pbc
        self.update_pos = update_pos
        self.dim = dim

        # in_features have to be of the same size as out_features for the time being
        self.egnn = torch.nn.ModuleList(modules=(EGNN_Layer(
            self.out_features, self.out_features, self.hidden_features, update_pos=self.update_pos
        ) for _ in range(self.hidden_layer)))

        self.head_pre_pool = nn.Sequential(nn.Linear(self.out_features, self.hidden_features),
                                           Swish(),
                                           nn.Linear(self.hidden_features, self.hidden_features))
        self.head_post_pool = nn.Sequential(nn.Linear(self.hidden_features, hidden_features),
                                            Swish(),
                                            nn.Linear(hidden_features, 1))

        self.embedding_mlp = nn.Sequential(nn.Linear(self.in_features, self.out_features),
                                           )

        # read atom map
        atom_map = torch.zeros(101, 9)
        for i in range(101):
            atom_map[i] = torch.tensor(CONTINUOUS_EMBEDDINGS[i])

        # normalize along each dimension
        atom_map[0] = np.nan
        atom_map_notnan = atom_map[atom_map[:, 0] == atom_map[:, 0]]
        atom_map_min = torch.min(atom_map_notnan, dim=0)[0]
        atom_map_max = torch.max(atom_map_notnan, dim=0)[0]
        atom_map_gap = atom_map_max - atom_map_min
        # squash to [0,1]
        atom_map = (atom_map - atom_map_min.view(1, -1)) / atom_map_gap.view(1, -1)
        self.atom_map = torch.nn.Parameter(atom_map, requires_grad=False)


    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        pos = data.pos
        batch = data.batch

        if self.otf_graph:
            edge_index, cell_offsets, neighbors = radius_graph_pbc(
                data, self.cutoff, 50, data.pos.device
            )
            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
            data.neighbors = neighbors

        if self.use_pbc:
            out = get_pbc_distances(
                pos,
                data.edge_index,
                data.cell,
                data.cell_offsets,
                data.neighbors,
                return_offsets=True,
            )

            edge_index = out["edge_index"]
            cell_offsets = out["offsets"]
        else:
            edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
            raise NotImplementedError

        h = self.atom_map[data.atomic_numbers.long()]

        """ Propagate messages along edges and average over energies"""
        h = self.embedding_mlp(h)
        for i in range(self.hidden_layer):
            h, pos = self.egnn[i](h, pos, edge_index, cell_offsets, batch)

        out = self.head_pre_pool(h)
        out = global_mean_pool(out, batch)
        energy = self.head_post_pool(out)

        return energy


    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, data):
        energy = self._forward(data)
        return energy

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


