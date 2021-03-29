"""
E (n) Equivariant Graph Neural Networks

Author: Johannes Brandstetter, Rob Hesselink
---

This code implements the paper
E (n) Equivariant Graph Neural Networks
by Victor Garcia Satorras, Emiel Hoogeboom, Max Welling
https://arxiv.org/abs/2102.09844
---
"""

import numpy as np
import torch
from torch import nn
from torch_scatter import scatter
from torch_geometric.nn.acts import swish
from torch_geometric.nn import MessagePassing

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

class Swish_(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

SiLU = nn.SiLU if hasattr(nn, 'SiLU') else Swish_


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
    def __init__(self, in_features, out_features, hidden_features, dim=3, update_pos=True):
        super(EGNN_Layer, self).__init__(node_dim=-2)
        self.update_pos = update_pos
        self.dim = dim
        self.message_net = nn.Sequential(nn.Linear(2*in_features + 1, hidden_features),
                                         SiLU(),
                                         nn.Linear(hidden_features, hidden_features)
                                         )
        self.update_net = nn.Sequential(nn.Linear(in_features + hidden_features, hidden_features),
                                        SiLU(),
                                        nn.Linear(hidden_features, out_features)
                                        )
        if self.update_pos:
            self.pos_net = nn.Sequential(nn.Linear(hidden_features, hidden_features),
                                         SiLU(),
                                         nn.Linear(hidden_features, dim)
                                         )

    def forward(self, x, pos, edge_index):
        """ Propagate messages along edges """
        x, pos = self.propagate(edge_index, x=x, pos=pos)
        return x, pos

    def message(self, x_i, x_j, pos_i, pos_j):
        """ Message according to eqs 3-4 in the paper """
        d = (pos_i - pos_j).pow(2).sum(-1, keepdims=True)
        message = self.message_net(torch.cat((x_i, x_j, d), dim=-1))
        if self.update_pos:
            pos_message = (pos_i - pos_j) * self.pos_net(message)
            # torch geometric does not support tuple outputs.
            message = torch.cat((pos_message, message), dim=-1)
        return message

    def update(self, message, x, pos):
        """ Update according to eq 6 in the paper """
        if self.update_pos:
            pos_message = message[:, :self.dim]
            message = message[:, self.dim:]
            pos += pos_message
        x = self.update_net(torch.cat((x, message), dim=-1))
        return x, pos


@registry.register_model("egnn")
class EGNN(torch.nn.Module):
    def __init__(
        self,
        num_atoms,  # not used
        bond_feat_dim,  # not used
        num_targets,  # not used
        in_features,
        out_features,
        hidden_features,
        hidden_layer=6,
        dim=3,
        update_pos=True,
        regress_forces=False,
        use_pbc=True,
        otf_graph=False,
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

        self.egnn = torch.nn.ModuleList(modules=(EGNN_Layer(
            self.in_features if _ == 0 else self.out_features, self.out_features, self.hidden_features
        ) for _ in range(self.hidden_layer)))

        self.energy_mlp = nn.Sequential(nn.Linear(self.out_features, self.out_features),
                                        SiLU(),
                                        nn.Linear(self.out_features, 1)
                                        )

        # read atom map and atom radii
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
        else:
            edge_index = radius_graph(pos, r=self.cutoff, batch=batch)

        h = self.atom_map[data.atomic_numbers.long()]

        """ Propagate messages along edges """
        for i in range(self.hidden_layer):
            h, pos = self.egnn[i](h, pos, edge_index)

        return h, pos

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, data):
        h, pos = self._forward(data)
        out = scatter(h, data.batch, dim=0, reduce="add")
        energy = self.energy_mlp(out)
        return energy

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


