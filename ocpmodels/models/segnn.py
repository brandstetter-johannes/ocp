"""
Steerable E (n) Equivariant Graph Neural Networks

Author: Johannes Brandstetter, Rob Hesselink, Erik Bekkers
"""

import numpy as np
import torch
from collections import OrderedDict
from torch import nn
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot_orthogonal
from e3nn.o3 import Irreps
from e3nn.o3 import Linear, spherical_harmonics, FullyConnectedTensorProduct
from e3nn.nn import Gate, BatchNorm

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


class SEGNNModel(nn.Module):
    def __init__(self, input_features=5, output_features=1, hidden_features=128, N=7, update_pos=False, recurrent=False,
                 lmax_h=2, lmax_pos=2):
        super(SEGNNModel, self).__init__()

        # The representations used in the model
        self.irreps_in = Irreps("{0}x0e".format(input_features))
        self.irreps_hidden = BalencedIrreps(lmax_h, hidden_features)
        self.irreps_out = Irreps("{0}x0e".format(output_features))
        self.irreps_rel_pos = Irreps.spherical_harmonics(lmax_pos)

        # The embedding layer
        self.layers = [
            SEGNN(self.irreps_in, self.irreps_hidden, self.irreps_rel_pos, self.irreps_hidden, update_pos=update_pos,
                  recurrent=recurrent)]
        # The intermediate layers
        for i in range(N-1):
            self.layers.append(SEGNN(self.irreps_hidden, self.irreps_hidden, self.irreps_rel_pos, self.irreps_hidden,
                                     update_pos=update_pos, recurrent=recurrent))
        # To ModuleList
        self.layers = nn.ModuleList(self.layers)
        # The output network (via trivial representations)
        self.head_pre_pool = nn.Sequential(LinearPlusSwishGate(self.irreps_hidden, self.irreps_hidden),
                                           LinearNoGate(self.irreps_hidden, self.irreps_hidden))
        self.head_post_pool = nn.Sequential(LinearPlusSwishGate(self.irreps_hidden, self.irreps_hidden),
                                            LinearNoGate(self.irreps_hidden, self.irreps_out))

    def forward(self, graph):
        x, pos = graph.x, graph.pos
        for layer in self.layers:
            x, pos = layer(x, pos, graph.edge_index)
        # Output head
        out = self.head_pre_pool(x)
        out = global_mean_pool(out, graph.batch)
        out = self.head_post_pool(out)

        return out


class SEGNN(MessagePassing):
    """E(N) equivariant message passing neural network.
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

    def __init__(self, irreps_in, irreps_out, irreps_rel_pos, irreps_hidden, dim=3, update_pos=False,
                 recurrent=False):
        super(SEGNN, self).__init__(node_dim=-2, aggr="mean")
        self.update_pos = update_pos
        self.dim = dim
        self.recurrent = recurrent
        self.irreps_rel_pos = irreps_rel_pos

        # Then this is the irrep that the message net takes as input
        # irreps_message_in = irreps_in + irreps_in + irreps_rel_pos
        # self.message_net = nn.Sequential(LinearPlusSwishGate(irreps_message_in, irreps_hidden),
        #                                   LinearPlusSwishGate(irreps_hidden, irreps_hidden))
        irreps_message_in = irreps_in + irreps_in + Irreps("1x0e")
        self.message_net = nn.Sequential(LinearPlusSwishGatePos(irreps_message_in, irreps_hidden, irreps_rel_pos),
                                         LinearPlusSwishGate(irreps_hidden, irreps_hidden))

        irreps_update_in = irreps_in + irreps_hidden
        self.update_net = nn.Sequential(LinearPlusSwishGate(irreps_update_in, irreps_hidden),
                                        LinearNoGate(irreps_hidden, irreps_out))

        if self.update_pos:  # TODO: currently not updated...
            hidden_features = 128
            self.pos_net = nn.Sequential(nn.Linear(hidden_features, hidden_features),
                                         Swish(),
                                         nn.Linear(hidden_features, dim))

        self.feature_norm = BatchNorm(irreps_out)

    def forward(self, x, pos, edge_index, cell_offsets):
        """ Propagate messages along edges """
        x, pos = self.propagate(edge_index, x=x, pos=pos, cell_offsets=cell_offsets)
        return x, pos

    def message(self, x_i, x_j, pos_i, pos_j, cell_offsets):
        """ Message according to eqs 3-4 in the paper """
        rel_pos = (pos_i - pos_j) + cell_offsets
        dist = rel_pos.norm(dim=-1, keepdim=True)
        rel_pos = spherical_harmonics(self.irreps_rel_pos, rel_pos, normalize=False, normalization='component')
        message = self.message_net(torch.cat((x_i, x_j, dist, rel_pos), dim=-1))

        if self.update_pos:  # TODO: currently no updated...
            pos_message = (pos_i - pos_j) * self.pos_net(message)
            # torch geometric does not support tuple outputs.
            message = torch.cat((pos_message, message), dim=-1)
        return message

    def update(self, message, x, pos):
        """ Update according to eq 6 in the paper """
        if self.update_pos:  # TODO: currently not updated...
            pos_message = message[:, :self.dim]
            message = message[:, self.dim:]

            pos += pos_message
        if self.recurrent:
            x += self.update_net(torch.cat((x, message), dim=-1))
        else:
            x = self.update_net(torch.cat((x, message), dim=-1))
        x = self.feature_norm(x)
        return x, pos

def BalencedIrreps(lmax, vec_dim):
    irrep_spec = "0e"
    for l in range(1,lmax + 1):
        irrep_spec += " + {0}e + {0}o".format(l)
    irrep_spec_split = irrep_spec.split(" + ")
    dims = [int(irrep[0]) * 2 + 1 for irrep in irrep_spec_split]
    # Compute ratios
    ratios = [1 / dim for dim in dims]
    # Determine how many copies per irrep
    irrep_copies = [int(vec_dim * r / len(ratios)) for r in ratios]
    # Determine the current effective irrep sizes
    irrep_dims = [n * dim for (n, dim) in zip(irrep_copies, dims)]
    # Add trivial irreps until the desired size is reached
    irrep_copies[0] += vec_dim - sum(irrep_dims)

    # Convert to string
    str_out = ''
    for (spec, dim) in zip(irrep_spec_split, irrep_copies):
        str_out += str(dim) + 'x' + spec
        str_out += ' + '
    str_out = str_out[:-3]
    # Generate the irrep
    return Irreps(str_out)

class LinearNoGate(torch.nn.Module):
    def __init__(self, irreps_in, irreps_out) -> None:
        super().__init__()

        # Build the layers
        self.linear_layer = FullyConnectedTensorProduct(
            irreps_in1=irreps_in,
            irreps_in2="1x0e",
            irreps_out=irreps_out, shared_weights=True)
        self.linear_layer_init()
        self.N_trivial = self.linear_layer.irreps_out[0].dim
        self.b = torch.nn.Parameter(torch.zeros(self.N_trivial, dtype=self.linear_layer.weight.dtype))

    def forward(self, data_in) -> torch.Tensor:
        # data_out = self.linear_layer(data_in)
        data_out = self.linear_layer(data_in, torch.ones_like(data_in[:,0:1]))
        data_out[:, :self.N_trivial] += self.b
        return data_out

    def linear_layer_init(self):
        with torch.no_grad():
            for weight in self.linear_layer.weight_views():
                mul_1, mul_2, mul_out = weight.shape
                # formula from torch.nn.init.xavier_uniform_
                a = (6 / (mul_1 * mul_2 + mul_out)) ** 0.5
                weight.data.uniform_(-a, a)

class LinearPlusSwishGate(torch.nn.Module):
    def __init__(self, irreps_in, irreps_out) -> None:
        super().__init__()

        # For the gate:
        # The first type is assumed to be scalar and passed through the activation
        irreps_g_scalars = Irreps(str(irreps_out[0]))
        # The remaining types are gated
        irreps_g_gate = Irreps("{}x0e".format(irreps_out.num_irreps - irreps_g_scalars.num_irreps))
        irreps_g_gated = Irreps(str(irreps_out[1:]))
        # so the gate needs the following irrep
        irreps_g = irreps_g_scalars + irreps_g_gate + irreps_g_gated

        # Build the layers
        # self.linear_layer = Linear(irreps_in, irreps_g)
        self.linear_layer = FullyConnectedTensorProduct(
            irreps_in1=irreps_in,
            irreps_in2="1x0e",
            irreps_out=irreps_g, shared_weights=True)
        self.linear_layer_init()
        self.N_trivial = self.linear_layer.irreps_out[0].dim
        self.b = torch.nn.Parameter(torch.zeros(self.N_trivial, dtype=self.linear_layer.weight.dtype))
        self.gate = Gate(irreps_g_scalars, [Swish()], irreps_g_gate, [torch.sigmoid], irreps_g_gated)

    def forward(self, data_in) -> torch.Tensor:
        # data_out = self.linear_layer(data_in)
        data_out = self.linear_layer(data_in, torch.ones_like(data_in[:,0:1]))
        data_out[:,:self.N_trivial] += self.b
        data_out = self.gate(data_out)
        return data_out

    def linear_layer_init(self):
        with torch.no_grad():
            for weight in self.linear_layer.weight_views():
                mul_1, mul_2, mul_out = weight.shape
                # formula from torch.nn.init.xavier_uniform_
                a = (6 / (mul_1 * mul_2 + mul_out)) ** 0.5
                weight.data.uniform_(-a, a)

class LinearPlusSwishGatePos(torch.nn.Module):
    def __init__(self, irreps_in, irreps_out, irreps_pos) -> None:
        super().__init__()
        self.irreps_pos = irreps_pos
        # For the gate:
        # The first type is assumed to be scalar and passed through the activation
        irreps_g_scalars = Irreps(str(irreps_out[0]))
        # The remaining types are gated
        irreps_g_gate = Irreps("{}x0e".format(irreps_out.num_irreps - irreps_g_scalars.num_irreps))
        irreps_g_gated = Irreps(str(irreps_out[1:]))
        # so the gate needs the following irrep
        irreps_g = irreps_g_scalars + irreps_g_gate + irreps_g_gated

        # Build the layers
        # self.linear_layer = Linear(irreps_in, irreps_g)
        self.linear_layer = FullyConnectedTensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=irreps_pos,
            irreps_out=irreps_g,
            shared_weights = True
        )
        self.linear_layer_init()
        self.N_trivial = self.linear_layer.irreps_out[0].dim
        self.b = torch.nn.Parameter(torch.zeros(self.N_trivial, dtype=self.linear_layer.weight.dtype))
        self.gate = Gate(irreps_g_scalars, [Swish()], irreps_g_gate, [torch.sigmoid], irreps_g_gated)

    def forward(self, data_in) -> torch.Tensor:
        data_in_hidden = data_in[:,:-self.irreps_pos.dim]
        data_in_pos = data_in[:, -self.irreps_pos.dim:]
        data_out = self.linear_layer(data_in_hidden, data_in_pos)
        data_out[:, :self.N_trivial] += self.b
        data_out = self.gate(data_out)
        return data_out

    def linear_layer_init(self):
        with torch.no_grad():
            for weight in self.linear_layer.weight_views():
                mul_1, mul_2, mul_out = weight.shape
                # formula from torch.nn.init.xavier_uniform_
                a = (6 / (mul_1 * mul_2 + mul_out)) ** 0.5
                weight.data.uniform_(-a, a)


@registry.register_model("segnn")
class SEGNNModel(torch.nn.Module):
    def __init__(
        self,
        num_atoms,  # not used
        bond_feat_dim,  # not used
        num_targets,  # not used
        in_features=9,
        out_features=1,
        hidden_features=256,
        hidden_layer=7,
        dim=3,
        lmax_h=2,
        lmax_pos=2,
        update_pos=False,
        recurrent=False,
        regress_forces=False,
        use_pbc=True,
        otf_graph=False
    ):

        super(SEGNNModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.hidden_layer = hidden_layer
        self.regress_forces = regress_forces
        self.otf_graph = otf_graph
        self.use_pbc = use_pbc
        self.update_pos = update_pos
        self.recurrent = recurrent
        self.dim = dim
        self.lmax_h = lmax_h
        self.lmax_pos = lmax_pos

        # The representations used in the model
        self.irreps_in = Irreps("{0}x0e".format(self.in_features))
        self.irreps_hidden = BalencedIrreps(lmax_h, self.hidden_features)
        self.irreps_out = Irreps("{0}x0e".format(self.out_features))
        self.irreps_rel_pos = Irreps.spherical_harmonics(self.lmax_pos)

        # The embedding layer
        self.layers = [
            SEGNN(self.irreps_in, self.irreps_hidden, self.irreps_rel_pos, self.irreps_hidden, update_pos=self.update_pos,
                  recurrent=self.recurrent)]
        # The intermediate layers
        for i in range(self.hidden_layer):
            self.layers.append(SEGNN(self.irreps_hidden, self.irreps_hidden, self.irreps_rel_pos, self.irreps_hidden,
                                     update_pos=update_pos, recurrent=recurrent))
        # To ModuleList
        self.layers = nn.ModuleList(self.layers)
        # The output network (via trivial representations)
        self.head_pre_pool = nn.Sequential(LinearPlusSwishGate(self.irreps_hidden, self.irreps_hidden),
                                           LinearNoGate(self.irreps_hidden, self.irreps_hidden))
        self.head_post_pool = nn.Sequential(LinearPlusSwishGate(self.irreps_hidden, self.irreps_hidden),
                                            LinearNoGate(self.irreps_hidden, self.irreps_out))


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

        self.embedding_mlp = nn.Sequential(nn.Linear(9, 5))


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
        h = self.embedding_mlp(h)

        """ Propagate messages along edges and output final energies"""
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index, cell_offsets)
        # Output head
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


