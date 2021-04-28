"""
Steerable E (n) Equivariant Graph Neural Networks

"""

import numpy as np
import torch
from collections import OrderedDict
from torch import nn
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing, global_mean_pool
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
        super(SEGNN, self).__init__(node_dim=-2, aggr="mean")  # <---- mean aggregation is important for node steering
        self.update_pos = update_pos
        self.dim = dim
        self.recurrent = recurrent
        self.irreps_rel_pos = irreps_rel_pos

        # Each layer within the message net is now steered via the rel_pos
        irreps_message_in = (irreps_in + irreps_in + Irreps("1x0e")).simplify()  # xi + xj + dist
        # self.message_net = nn.Sequential(O3LinearSwishGate(irreps_message_in, irreps_hidden, irreps_rel_pos),
        #                                  O3LinearSwishGate(irreps_hidden, irreps_hidden))
        self.message_layer_1 = O3LinearSwishGate(irreps_message_in, irreps_hidden, irreps_rel_pos)
        self.message_layer_2 = O3LinearSwishGate(irreps_hidden, irreps_out, irreps_rel_pos)

        # Each layer within the update net is now also steered via a distribution on the sphere by taking the average
        # over all neighbor rel_pos of the to-be-updated node
        irreps_update_in = (irreps_in + irreps_hidden).simplify()
        # self.update_net = nn.Sequential(O3LinearSwishGate(irreps_update_in, irreps_hidden, irreps_rel_pos),
        #                                 O3Linear(irreps_hidden, irreps_out))
        self.update_layer_1 = O3LinearSwishGate(irreps_update_in, irreps_hidden, irreps_rel_pos)
        self.update_layer_2 = O3Linear(irreps_hidden, irreps_out, irreps_rel_pos)

        if self.update_pos:  # TODO: currently not updated...
            hidden_features = 128
            self.pos_net = nn.Sequential(nn.Linear(hidden_features, hidden_features),
                                         Swish(),
                                         nn.Linear(hidden_features, dim))

        #self.feature_norm = BatchNorm(irreps_hidden)


    def forward(self, x, pos, edge_index, cell_offsets):
        """ Propagate messages along edges """
        x, pos = self.propagate(edge_index, x=x, pos=pos, cell_offsets=cell_offsets)
        # x = self.feature_norm(x)
        return x, pos

    def message(self, x_i, x_j, pos_i, pos_j, cell_offsets):
        """ Message according to eqs 3-4 in the paper """
        rel_pos = (pos_i - pos_j) + cell_offsets
        dist = rel_pos.pow(2).sum(-1, keepdims=True)
        rel_pos = spherical_harmonics(self.irreps_rel_pos, rel_pos, normalize=True, normalization='component')
        # message = self.message_net(torch.cat((x_i, x_j, dist, rel_pos), dim=-1))
        message = self.message_layer_1(torch.cat((x_i, x_j, dist, rel_pos), dim=-1))
        message = self.message_layer_2(torch.cat((message, rel_pos), dim=-1))
        message = torch.cat((message, rel_pos), dim=-1) # <---- pass the relative position along
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

        rel_pos_data = message[:, -self.irreps_rel_pos.dim:]  # <---- extract the relative pos data
        update = self.update_layer_1(torch.cat((x, message), dim=-1))
        update = self.update_layer_2(torch.cat((update, rel_pos_data), dim=-1))  # <---- insert rel_pos data again
        if self.recurrent:
            # x += self.update_net(torch.cat((x, message), dim=-1))
            x += update
        else:
            # x = self.update_net(torch.cat((x, message), dim=-1))
            x = update

        return x, pos


def BalancedIrreps(lmax, vec_dim, sh_type = True):
    irrep_spec = "0e"
    for l in range(1, lmax + 1):
        if sh_type:
            irrep_spec +=  " + {0}".format(l) + ('e' if ( l % 2) == 0 else 'o')
        else:
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


class O3Linear(torch.nn.Module):
    def __init__(self, irreps_in1, irreps_out, irreps_in2=None, tp_rescale=True) -> None:
        super().__init__()

        self.irreps_in1 = irreps_in1
        self.irreps_out = irreps_out
        # Init irreps_in2
        if irreps_in2 == None:
            self.irreps_in2_provided = False
            self.irreps_in2 = Irreps("1x0e")
        else:
            self.irreps_in2_provided = True
            self.irreps_in2 = irreps_in2
        self.tp_rescale = tp_rescale

        # Build the layers
        self.linear_layer = FullyConnectedTensorProduct(
            irreps_in1=self.irreps_in1,
            irreps_in2=self.irreps_in2,
            irreps_out=self.irreps_out, shared_weights=True, normalization='component')

        # For each zeroth order output irrep we need a bias
        # So first determine the order for each output tensor and their dims
        self.irreps_out_orders = [int(irrep_str[-2]) for irrep_str in str(self.irreps_out).split('+')]
        self.irreps_out_dims = [int(irrep_str.split('x')[0]) for irrep_str in str(self.irreps_out).split('+')]
        self.irreps_out_slices = self.irreps_out.slices()
        # Store tuples of slices and corresponding biaes in a list
        self.biases = []
        self.biases_slices = []
        self.biases_slice_idx = []
        for slice_idx in range(len(self.irreps_out_orders)):
            if self.irreps_out_orders[slice_idx] == 0:
                out_slice = self.irreps_out.slices()[slice_idx]
                out_bias = torch.nn.Parameter(
                    torch.zeros(self.irreps_out_dims[slice_idx], dtype=self.linear_layer.weight.dtype))
                self.biases += [out_bias]
                self.biases_slices += [out_slice]
                self.biases_slice_idx += [slice_idx]
        self.biases = torch.nn.ParameterList(self.biases)

        # Initialize the correction factors
        self.slices_sqrt_k = {}

        # Initialize similar to the torch.nn.Linear
        self.linear_layer_init()

    def linear_layer_init(self) -> None:
        with torch.no_grad():
            # Determine fan_in for each slice, it could be that each output slice is updated via several instructions
            slices_fan_in = {} # fan_in per slice
            for weight, instr in zip(self.linear_layer.weight_views(), self.linear_layer.instructions):
                slice_idx = instr[2]
                mul_1, mul_2, mul_out = weight.shape
                fan_in = mul_1 * mul_2
                slices_fan_in[slice_idx] = (slices_fan_in[slice_idx] + fan_in if slice_idx in slices_fan_in.keys() else fan_in)

            # Do the initialization of the weights in each instruction
            for weight, instr in zip(self.linear_layer.weight_views(), self.linear_layer.instructions):
                # The tensor product in e3nn already normalizes proportional to 1 / sqrt(fan_in), and the weights are by
                # default initialized with unif(-1,1). However, we want to be consistent with torch.nn.Linear and
                # initialize the weights with unif(-sqrt(k),sqrt(k)), with k = 1 / fan_in
                if self.tp_rescale:
                    sqrt_k = 1 / slices_fan_in[slice_idx] ** 0.5
                else:
                    sqrt_k = 1.
                weight.data.uniform_(-sqrt_k, sqrt_k)
                self.slices_sqrt_k[slice_idx] = (self.irreps_out_slices[slice_idx], sqrt_k)

            # Initialize the biases
            for (out_slice_idx, out_slice, out_bias) in zip(self.biases_slice_idx, self.biases_slices, self.biases):
                sqrt_k = 1 / slices_fan_in[out_slice_idx] ** 0.5
                out_bias.uniform_(-sqrt_k, sqrt_k)

    def forward_tp_rescale_bias(self, data_in) -> torch.Tensor:
        # Split based on irreps_in1 and irreps_in2
        if self.irreps_in2_provided:
            data_in1 = data_in[:, :-self.irreps_in2.dim]
            data_in2 = data_in[:, -self.irreps_in2.dim:]
        else:
            data_in1 = data_in
            data_in2 = torch.ones_like(data_in[:, 0:1])
        data_out = self.linear_layer(data_in1, data_in2)
        # Apply corrections
        if self.tp_rescale:
            for (slice, slice_sqrt_k) in self.slices_sqrt_k.values():
                data_out[:,slice] /= slice_sqrt_k
        # Add the biases
        for (_, slice, bias) in zip(self.biases_slice_idx, self.biases_slices, self.biases):
            data_out[:,slice] += bias
        # Return result
        return data_out

    def forward(self, data_in) -> torch.Tensor:
        # Apply the tensor product, the rescaling and the bias
        data_out = self.forward_tp_rescale_bias(data_in)
        return data_out


class O3LinearSwishGate(O3Linear):
    def __init__(self, irreps_in1, irreps_out, irreps_in2 = None) -> None:
        # For the gate the output of the linear needs to have an extra number of scalar irreps equal to the amount of
        # non scalar irreps:
        # The first type is assumed to be scalar and passed through the activation
        irreps_g_scalars = Irreps(str(irreps_out[0]))
        # The remaining types are gated
        irreps_g_gate = Irreps("{}x0e".format(irreps_out.num_irreps - irreps_g_scalars.num_irreps))
        irreps_g_gated = Irreps(str(irreps_out[1:]))
        # So the gate needs the following irrep as input, this is the output irrep of the tensor product
        irreps_g = (irreps_g_scalars + irreps_g_gate + irreps_g_gated).simplify()

        # Build the layers
        super(O3LinearSwishGate, self).__init__(irreps_in1, irreps_g, irreps_in2)
        if irreps_g_gated.num_irreps > 0:
            self.gate = Gate(irreps_g_scalars, [Swish()], irreps_g_gate, [torch.sigmoid], irreps_g_gated)
        else:
            self.gate = Swish()

    def forward(self, data_in) -> torch.Tensor:
        # Apply the tensor product, the rescaling and the bias
        data_out = self.forward_tp_rescale_bias(data_in)
        # Apply the gate
        data_out = self.gate(data_out)
        # Return result
        return data_out



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
        N=7,
        dim=3,
        lmax_h=2,
        lmax_pos=2,
        update_pos=False,
        recurrent=True,
        regress_forces=False,
        use_pbc=True,
        otf_graph=False
    ):

        super(SEGNNModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.N = N
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
        self.irreps_hidden = BalancedIrreps(self.lmax_h, self.hidden_features)
        self.irreps_hidden_scalar = Irreps("{0}x0e".format(self.hidden_features))
        self.irreps_out = Irreps("{0}x0e".format(self.out_features))
        self.irreps_rel_pos = Irreps.spherical_harmonics(self.lmax_pos)

        # The embedding layer (acts point-wise, no orientation information so only use trivial/scalar irreps)
        self.embedding = nn.Sequential(O3LinearSwishGate(self.irreps_in, self.irreps_hidden_scalar),
                                       O3Linear(self.irreps_hidden_scalar, self.irreps_hidden_scalar))
        # The intermediate layers
        self.layers = []
        # The first layer changes from scalar irreps to irreps of some max order (lmax_h)
        self.layers.append(SEGNN(self.irreps_hidden_scalar, self.irreps_hidden, self.irreps_rel_pos, self.irreps_hidden,
                                 update_pos=self.update_pos, recurrent=False))
        # Subsequent layers act on the irreps of some max order (lmax_h)
        for i in range(self.N - 2):
            self.layers.append(SEGNN(self.irreps_hidden, self.irreps_hidden, self.irreps_rel_pos, self.irreps_hidden,
                                     update_pos=self.update_pos, recurrent=self.recurrent))
        # The last layer of the SEGNN block converts back to scalar irreps
        self.layers.append(
            SEGNN(self.irreps_hidden, self.irreps_hidden_scalar, self.irreps_rel_pos, self.irreps_hidden_scalar,
                  update_pos=self.update_pos, recurrent=False))
        # To ModuleList
        self.layers = nn.ModuleList(self.layers)

        # The output network (again via point-wise operation via scalar irreps)
        self.head_pre_pool = nn.Sequential(O3LinearSwishGate(self.irreps_hidden_scalar, self.irreps_hidden_scalar),
                                           O3Linear(self.irreps_hidden_scalar, self.irreps_hidden_scalar))
        self.head_post_pool = nn.Sequential(O3LinearSwishGate(self.irreps_hidden_scalar, self.irreps_hidden_scalar),
                                            O3Linear(self.irreps_hidden_scalar, self.irreps_out))


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
        h = self.embedding(h)
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index, cell_offsets)
        # Output heads
        h = self.head_pre_pool(h)
        h = global_mean_pool(h, batch)
        energy = self.head_post_pool(h)

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


