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
from torch_geometric.utils import remove_isolated_nodes
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
    """
        Computes the node and edge attributes based on relative positions
    """

    def __init__(self, node_in_irreps, node_hidden_irreps, node_out_irreps, attr_irreps, update_pos=False,
                 recurrent=True, infer_edges=False, edge_weight=False):
        super(SEGNN, self).__init__(node_dim=-2, aggr="add")

        self.update_pos = update_pos
        self.recurrent = recurrent
        self.infer_edges = infer_edges
        self.edge_weight = edge_weight

        # The message network layers
        irreps_message_in = (node_in_irreps + node_in_irreps + Irreps("1x0e")).simplify()
        self.message_layer_1 = O3TensorProductSwishGate(irreps_message_in,
                                                        node_hidden_irreps,
                                                        attr_irreps)
        self.message_layer_2 = O3TensorProductSwishGate(node_hidden_irreps,
                                                        node_hidden_irreps,
                                                        attr_irreps)

        # The node update layers
        irreps_update_in = (node_in_irreps + node_hidden_irreps).simplify()
        self.update_layer_1 = O3TensorProductSwishGate(irreps_update_in,
                                                       node_hidden_irreps,
                                                       attr_irreps)
        self.update_layer_2 = O3TensorProduct(node_hidden_irreps,
                                              node_out_irreps,
                                              attr_irreps)

        # Position update network
        if self.update_pos:  # TODO: currently not updated...
            self.pos_update_layer_1 = None  # O3TensorProductSwishGate
            self.pos_update_layer_2 = None  # O3TensorProduct

        if self.infer_edges:
            self.inf_net_1 = O3TensorProduct(node_hidden_irreps, Irreps("1x0e"), attr_irreps)
            self.inf_net_2 = nn.Sigmoid()

        # self.feature_norm = BatchNorm(node_out_irreps)
        # self.message_norm = BatchNorm(node_out_irreps)


    def forward(self, x, pos, edge_index, edge_dist, edge_attr, node_attr):
        """ Propagate messages along edges """
        x, pos = self.propagate(edge_index, x=x, pos=pos, edge_dist=edge_dist, node_attr=node_attr, edge_attr=edge_attr) # TODO: continue here!
        # x = self.feature_norm(x)
        return x, pos

    def message(self, x_i, x_j, edge_dist, edge_attr):
        """ Message according to eqs 3-4 in the paper """
        message = self.message_layer_1(torch.cat((x_i, x_j, edge_dist), dim=-1), edge_attr)
        message = self.message_layer_2(message, edge_attr)
        # message = self.message_norm(message)
        if self.update_pos:
            pos_message = None
        if self.infer_edges:
            message = self.inf_net_2(self.inf_net_1(message, edge_attr)) * message
        if self.edge_weight:
            edge_weight = torch.cos(0.5 * torch.sqrt(edge_dist) * 3.14 / 6.)
            message = message * edge_weight.view(-1, 1)

        return message

    def update(self, message, x, pos, node_attr):
        """ Update according to eq 6 in the paper """
        if self.update_pos:  # TODO: currently not updated...
            pos = pos

        update = self.update_layer_1(torch.cat((x, message), dim=-1), node_attr)
        update = self.update_layer_2(update, node_attr)
        if self.recurrent:
            x += update
        else:
            x = update
        return x, pos


class NodeAttributeNetwork(MessagePassing):
    """
        Computes the node and edge attributes based on relative positions
    """

    def __init__(self):
        super(NodeAttributeNetwork, self).__init__(node_dim=-2, aggr="mean")  # <---- Mean of all edge features

    def forward(self, edge_index, edge_attr):
        """ Simply sums the edge attributes """
        node_attr = self.propagate(edge_index, edge_attr=edge_attr) # TODO: continue here!
        return node_attr

    def message(self, edge_attr):
        """ The message is the edge attribute """
        return edge_attr

    def update(self, node_attr):
        """ The input to update is the aggregated messages, and thus the node attribute """
        return node_attr


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
    #print('Determined irrep type:', str_out)
    return Irreps(str_out)

def WeightBalancedIrreps(irreps_in1_scalar, irreps_in2, sh = True, lmax=None):
    """
    Determines an irreps_in1 type of order irreps_in2.lmax that when used in a tensor product
    irreps_in1 x irreps_in2 -> irreps_in1
    would have the same number of weights as for a standard linear layer, e.g. a tensor product
    irreps_in1_scalar x "1x0e" -> irreps_in1_scaler
    """
    n = 1
    if (lmax == None):
        lmax = irreps_in2.lmax
    irreps_in1 = (Irreps.spherical_harmonics(lmax) * n).sort().irreps.simplify() if sh else BalancedIrreps(lmax, n)
    weight_numel1 = FullyConnectedTensorProduct(irreps_in1, irreps_in2, irreps_in1).weight_numel
    weight_numel_scalar = FullyConnectedTensorProduct(irreps_in1_scalar, Irreps("1x0e"), irreps_in1_scalar).weight_numel
    while weight_numel1 < weight_numel_scalar:  # TODO: somewhat suboptimal implementation...
        n += 1
        irreps_in1 = (Irreps.spherical_harmonics(lmax) * n).sort().irreps.simplify() if sh else BalancedIrreps(lmax, n)
        weight_numel1 = FullyConnectedTensorProduct(irreps_in1, irreps_in2, irreps_in1).weight_numel
    print('Determined irrep type:', irreps_in1)

    return Irreps(irreps_in1)


class O3TensorProduct(torch.nn.Module):
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
        self.tp = FullyConnectedTensorProduct(
            irreps_in1=self.irreps_in1,
            irreps_in2=self.irreps_in2,
            irreps_out=self.irreps_out, shared_weights=True, normalization='component')

        # For each zeroth order output irrep we need a bias
        # So first determine the order for each output tensor and their dims
        self.irreps_out_orders = [int(irrep_str[-2]) for irrep_str in str(irreps_out).split('+')]
        self.irreps_out_dims = [int(irrep_str.split('x')[0]) for irrep_str in str(irreps_out).split('+')]
        self.irreps_out_slices = irreps_out.slices()
        # Store tuples of slices and corresponding biases in a list
        self.biases = []
        self.biases_slices = []
        self.biases_slice_idx = []
        for slice_idx in range(len(self.irreps_out_orders)):
            if self.irreps_out_orders[slice_idx] == 0:
                out_slice = irreps_out.slices()[slice_idx]
                out_bias = torch.nn.Parameter(
                    torch.zeros(self.irreps_out_dims[slice_idx], dtype=self.tp.weight.dtype))
                self.biases += [out_bias]
                self.biases_slices += [out_slice]
                self.biases_slice_idx += [slice_idx]
        self.biases = torch.nn.ParameterList(self.biases)

        # Initialize the correction factors
        self.slices_sqrt_k = {}

        # Initialize similar to the torch.nn.Linear
        self.tensor_product_init()

    def tensor_product_init(self) -> None:
        with torch.no_grad():
            # Determine fan_in for each slice, it could be that each output slice is updated via several instructions
            slices_fan_in = {} # fan_in per slice
            for weight, instr in zip(self.tp.weight_views(), self.tp.instructions):
                slice_idx = instr[2]
                mul_1, mul_2, mul_out = weight.shape
                fan_in = mul_1 * mul_2
                slices_fan_in[slice_idx] = (slices_fan_in[slice_idx] + fan_in if slice_idx in slices_fan_in.keys() else fan_in)

            # Do the initialization of the weights in each instruction
            for weight, instr in zip(self.tp.weight_views(), self.tp.instructions):
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

    def forward_tp_rescale_bias(self, data_in1, data_in2=None) -> torch.Tensor:
        if data_in2 == None:
            data_in2 = torch.ones_like(data_in1[:, 0:1])

        data_out = self.tp(data_in1, data_in2)
        # Apply corrections
        if self.tp_rescale:
            for (slice, slice_sqrt_k) in self.slices_sqrt_k.values():
                data_out[:,slice] /= slice_sqrt_k
        # Add the biases
        for (_, slice, bias) in zip(self.biases_slice_idx, self.biases_slices, self.biases):
            data_out[:,slice] += bias
        # Return result
        return data_out

    def forward(self, data_in1, data_in2=None) -> torch.Tensor:
        # Apply the tensor product, the rescaling and the bias
        data_out = self.forward_tp_rescale_bias(data_in1, data_in2)
        return data_out


class O3TensorProductSwishGate(O3TensorProduct):
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
        super(O3TensorProductSwishGate, self).__init__(irreps_in1, irreps_g, irreps_in2)
        if irreps_g_gated.num_irreps > 0:
            self.gate = Gate(irreps_g_scalars, [Swish()], irreps_g_gate, [torch.sigmoid], irreps_g_gated)
        else:
            self.gate = Swish()

    def forward(self, data_in1, data_in2=None) -> torch.Tensor:
        # Apply the tensor product, the rescaling and the bias
        data_out = self.forward_tp_rescale_bias(data_in1, data_in2)
        # Apply the gate
        data_out = self.gate(data_out)
        # Return result
        return data_out



@registry.register_model("segnn2")
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
        lmax_pos=None,
        update_pos=False,
        infer_edges=False,
        edge_weight=False,
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
        if(lmax_pos == None):
            self.lmax_pos = self.lmax_h
        self.infer_edges = infer_edges
        self.edge_weight = edge_weight

        # Irreps for the node features
        node_in_irreps_scalar = Irreps("{0}x0e".format(self.in_features))  # This is the type of the input
        #node_hidden_irreps = BalancedIrreps(self.lmax_h, self.hidden_features)  # This is the type on the hidden reps
        node_hidden_irreps_scalar = Irreps("{0}x0e".format(self.hidden_features))  # For the output layers
        node_out_irreps_scalar = Irreps("{0}x0e".format(self.out_features))  # This is the type on the output

        # Irreps for the edge and node attributes
        attr_irreps = Irreps.spherical_harmonics(self.lmax_pos)
        self.attr_irreps = attr_irreps
        print('Determined attr irrep type:', self.attr_irreps)

        node_hidden_irreps = WeightBalancedIrreps(node_hidden_irreps_scalar, attr_irreps, False, lmax=self.lmax_h)  # True: copies of sh

        # Network for computing the node attributes
        self.node_attribute_net = NodeAttributeNetwork()

        # The embedding layer (acts point-wise, no orientation information so only use trivial/scalar irreps)
        self.embedding_layer_1 = O3TensorProductSwishGate(node_in_irreps_scalar,  # in
                                                          node_hidden_irreps,  # out
                                                          attr_irreps)  # steerable attribute
        self.embedding_layer_2 = O3TensorProductSwishGate(node_hidden_irreps,  # in
                                                          node_hidden_irreps,  # out
                                                          attr_irreps)  # steerable attribute
        self.embedding_layer_3 = O3TensorProduct(node_hidden_irreps,  # in
                                                 node_hidden_irreps,  # out
                                                 attr_irreps)  # steerable attribute

        # The main layers
        self.layers = []
        for i in range(self.N):
            self.layers.append(SEGNN(node_hidden_irreps,  # in
                                     node_hidden_irreps,  # hidden
                                     node_hidden_irreps,  # out
                                     attr_irreps,  # steerable attribute
                                     update_pos=self.update_pos,
                                     recurrent=self.recurrent,
                                     infer_edges=self.infer_edges,
                                     edge_weight=self.edge_weight))
        self.layers = nn.ModuleList(self.layers)

        # The output network (again via point-wise operation via scalar irreps)
        self.head_pre_pool_layer_1 = O3TensorProductSwishGate(node_hidden_irreps,  # in
                                                              node_hidden_irreps_scalar,  # out
                                                              attr_irreps)  # steerable attribute
        self.head_pre_pool_layer_2 = O3TensorProduct(node_hidden_irreps_scalar,  # in
                                                     node_hidden_irreps_scalar)  # out
        self.head_post_pool_layer_1 = O3TensorProductSwishGate(node_hidden_irreps_scalar,  # in
                                                               node_hidden_irreps_scalar)  # out
        self.head_post_pool_layer_2 = O3TensorProduct(node_hidden_irreps_scalar,  # in
                                                      node_out_irreps_scalar)  # out


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

        # construct the node and edge attributes
        rel_pos = (pos[edge_index[0]] - pos[edge_index[1]]) + cell_offsets
        edge_dist = rel_pos.pow(2).sum(-1, keepdims=True)
        edge_attr = spherical_harmonics(self.attr_irreps, rel_pos, normalize=True, normalization='component')
        node_attr = self.node_attribute_net(edge_index, edge_attr)
        if (data.contains_isolated_nodes() and edge_index.max().item() + 1 != data.num_nodes):
            nr_add_attr = data.num_nodes - (edge_index.max().item() + 1)
            add_attr = node_attr.new_tensor(np.tile(np.eye(node_attr.shape[-1])[0,:], (nr_add_attr,1)))
            #add_attr = node_attr.new_tensor(np.zeros((nr_add_attr, node_attr.shape[-1])))
            node_attr = torch.cat((node_attr, add_attr), -2)

        # node_attr, edge_attr = self.attribute_net(pos, edge_index)
        x = self.atom_map[data.atomic_numbers.long()]
        x = self.embedding_layer_1(x, node_attr)
        x = self.embedding_layer_2(x, node_attr)
        x = self.embedding_layer_3(x, node_attr)

        # The main layers
        for layer in self.layers:
            x, pos = layer(x, pos, edge_index, edge_dist, edge_attr, node_attr)

        # Output head
        x = self.head_pre_pool_layer_1(x, node_attr)
        x = self.head_pre_pool_layer_2(x)
        x = global_mean_pool(x, batch)
        x = self.head_post_pool_layer_1(x)
        x = self.head_post_pool_layer_2(x)

        # Return the result
        return x



    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, data):
        energy = self._forward(data)
        return energy

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


