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
from collections import OrderedDict
from torch import nn
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing, InstanceNorm
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


def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def add_phantom_atoms(h, pos, edge_index, cell_offsets, tags, batch):
    nr_batches = len(torch.unique(batch))
    # add features and positions of phantom atoms
    h_new = h.clone()
    pos_new = pos.clone()
    for b in range(nr_batches):
        h_add = scatter(h[batch == b], tags[batch == b], dim=0, reduce="mean")
        h_new = torch.cat((h_new, h_add), 0)
        pos_add = scatter(pos[batch == b], tags[batch == b], dim=0, reduce="mean")
        pos_new = torch.cat((pos_new, pos_add), 0)
    # add aggregating nodes at end
    for b in range(nr_batches):
        h_add = scatter(h[batch == b], tags[batch == b], dim=0, reduce="mean")
        h_agg = h_add.mean(0)[None, :]
        h_new = torch.cat((h_new, h_agg), 0)
        pos_add = scatter(pos[batch == b], tags[batch == b], dim=0, reduce="mean")
        pos_agg = pos_add.mean(0)[None, :]
        pos_new = torch.cat((pos_new, pos_agg), 0)
    # initialize new edges
    edge_i = edge_index.new_zeros((0,))
    edge_j = edge_index.new_tensor(np.arange(0, len(batch)))
    # help counter for phantom atoms and aggregating phantom atoms
    ac = 0
    ag = 0
    for b in range(nr_batches):
        # categories in one batch
        n0 = (torch.bincount(tags[batch == b])[0]).item()
        n1 = (torch.bincount(tags[batch == b])[1]).item()
        n2 = (torch.bincount(tags[batch == b])[2]).item()
        # add indices of new phantom atoms
        ex0 = edge_index.new_tensor(np.full((n0,), (len(batch) + ac)))
        ex1 = edge_index.new_tensor(np.full((n1,), (len(batch) + ac + 1)))
        ex2 = edge_index.new_tensor(np.full((n2,), (len(batch) + ac + 2)))
        # concate new edges
        edge_i = torch.cat((edge_i, ex0, ex1, ex2), 0)
        # raise help counter
        ac += 3
    # add edges of aggregating atom
    for b in range(nr_batches):
        ex_agg = edge_index.new_tensor(np.full((3,), (len(batch) + ac)))
        p = edge_index.new_tensor(
            np.array([len(batch) + ag, len(batch) + ag + 1, len(batch) + ag + 2]))
        # concate aggregating edges
        edge_i = torch.cat((edge_i, ex_agg), 0)
        edge_j = torch.cat((edge_j, p), 0)
        # raise help counters
        ac += 1
        ag += 3
    # add new edges to adjacency
    suffix = torch.cat((torch.cat((edge_i, edge_j)), torch.cat((edge_j, edge_i)))).view(2, -1)
    edge_index_new = edge_index.clone()
    edge_index_new = torch.cat((edge_index_new, suffix), 1)
    # add 0 cell offsets for phantom atoms and aggregating atoms
    suffix_cell_offsets = cell_offsets.new_tensor(np.zeros((len(batch) * 2 + nr_batches * 6, 3)))
    cell_offsets_new = torch.cat((cell_offsets, suffix_cell_offsets), 0)
    # add tag==2 to all phantom atoms and aggregating atoms
    suffix_tags = tags.new_tensor(np.full((nr_batches * 4,), 2))
    tags_new = torch.cat((tags, suffix_tags), 0)
    # only adsorbat atoms and phantom atoms are allowed to move
    tags_new[tags_new != 2] = 0
    tags_new[tags_new == 2] = 1
    tags_new = tags_new[:, None]

    return h_new, pos_new, edge_index_new, cell_offsets_new, tags_new


def add_phantom_atom(h, pos, edge_index, cell_offsets, tags, batch):
    # add features of phantom atom
    h_suffix = scatter(h, batch, dim=0, reduce="mean")
    h_new = torch.cat((h, h_suffix), 0)
    # add position of phantom atom
    pos_suffix = scatter(pos, batch, dim=0, reduce="mean")
    pos_new = torch.cat((pos, pos_suffix), 0)
    # add phantom atom to adjacency matrix, add (0,0,0) to cell_offsets
    edge_i = edge_index.new_zeros((0,))
    edge_j = edge_index.new_tensor(np.arange(0, len(batch)))
    nr_batches = len(torch.unique(batch))
    for batch_nr in range(nr_batches):
        # count entries of the same number in batch
        nr_entries = (torch.bincount(batch)[batch_nr]).item()
        # add nr_entries times new number (phantom atom) for each batch
        bi = edge_index.new_tensor(np.full((nr_entries,), (len(batch) + batch_nr)))
        edge_i = torch.cat((edge_i, bi), 0)
    suffix = torch.cat((torch.cat((edge_i, edge_j)), torch.cat((edge_j, edge_i)))).view(2,-1)
    # connections of phantom add to all atoms and from all atoms
    edge_index_new = edge_index.clone()
    edge_index_new = torch.cat((edge_index_new, suffix), 1)
    # add cell offsets
    suffix_cell_offsets = cell_offsets.new_tensor(np.zeros((len(batch) * 2, 3)))
    cell_offsets_new = torch.cat((cell_offsets, suffix_cell_offsets), 0)
    # update tags (if atom is allowed to move) -> we only want adsorbat atoms to move
    suffix_tags = tags.new_tensor(np.full((nr_batches,), 2))
    tags_new = torch.cat((tags, suffix_tags), 0)
    tags_new[tags_new != 2] = 0
    tags_new[tags_new == 2] = 1
    tags_new = tags_new[:, None]
    # update batch indices for phantom atom
    suffix_batch = batch.new_tensor(np.arange(0, nr_batches))
    batch_new = torch.cat((batch, suffix_batch), 0)

    return h_new, pos_new, edge_index_new, cell_offsets_new, tags_new, batch_new


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

        #self.ln = nn.LayerNorm(hidden_features)
        self.message_net = nn.Sequential(nn.Linear(2*in_features + 1, hidden_features),
                                        Swish(),
                                        nn.BatchNorm1d(hidden_features), #new
                                        nn.Linear(hidden_features, out_features),
                                        Swish() #new
                                        )
        self.update_net = nn.Sequential(nn.Linear(in_features + hidden_features, hidden_features),
                                        Swish(),
                                        nn.Linear(hidden_features, out_features),
                                        Swish() #new
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

    def forward(self, x, pos, edge_index, cell_offsets, tags, batch):
        """ Propagate messages along edges """
        x, pos = self.propagate(edge_index, x=x, pos=pos, cell_offsets=cell_offsets, tags=tags)
        x = self.norm(x, batch) # new
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

    def update(self, message, x, pos, tags):
        """ Update according to eq 6 in the paper """
        pos_message = message[:, :self.dim]
        message = message[:, self.dim:]
        if self.update_pos:
            pos += pos_message*tags
        #x += self.update_net(torch.cat((x, message), dim=-1))
        update = self.update_net(torch.cat((x, message), dim=-1)) #new
        x += update
        #x = torch.nn.functional.softplus(self.ln(update) + x) #new
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
        #self.egnn = torch.nn.ModuleList(modules=(EGNN_Layer(
        #    self.in_features if _ == 0 else self.out_features, self.out_features, self.hidden_features, update_pos=self.update_pos
        #) for _ in range(self.hidden_layer)))

        self.energy_mlp = nn.Sequential(nn.Linear(self.out_features, 512),
                                        Swish(),
                                        nn.Linear(512, 128),
                                        Swish(),
                                        nn.Linear(128, 1)
                                        )

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

        # add phantom atom with has average features and connections to all other atoms
        h, pos, edge_index, cell_offsets, tags, batch_new = add_phantom_atom(h,
                                                                        pos,
                                                                        edge_index,
                                                                        cell_offsets,
                                                                        data.tags,
                                                                        batch)

        """ Propagate messages along edges and average over energies"""
        h = self.embedding_mlp(h)
        for i in range(self.hidden_layer):
            h, pos = self.egnn[i](h, pos, edge_index, cell_offsets, tags, batch_new)

        energy = self.energy_mlp(h[len(batch):])
        #nr_batches = len(torch.unique(batch))
        #energy = self.energy_mlp(h[len(batch) + 3 * nr_batches:])
        #out = torch.cat((h[len(batch):len(batch) + 1 * nr_batches],
        #                 h[len(batch) + 1 * nr_batches:len(batch) + 2 * nr_batches],
        #                 h[len(batch) + 2 * nr_batches:len(batch) + 3 * nr_batches],
        #                 h[len(batch) + 3 * nr_batches:]), 1)

        #energy = self.energy_mlp(out)
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


