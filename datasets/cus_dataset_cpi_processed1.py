import os
import torch
import dgl
import torch_geometric
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector, get_atom_feature_dims, \
    get_bond_feature_dims
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from scipy.constants import physical_constants


from commons.spherical_encoding import dist_emb

hartree2eV = physical_constants['hartree-electron volt relationship'][0]
###########################################################################
import biotite.structure as struc
import biotite.structure.io as strucio

class CusDatasetCPI(Dataset):

    def __init__(self, device='cuda:0', processed_fold=None, **kwargs):

        self.device = device 

        data_dict = torch.load(processed_fold)

        self.data = data_dict# self.meta_dict = {k: data_dict[k] for k in ('mol_id', 'edge_slices', 'atom_slices', 'n_atoms')}

    def mol_2_graph(self, idx):
        e_start = self.data['edge_slices'][idx]
        e_end = self.data['edge_slices'][idx + 1]
        start = self.data['atom_slices'][idx]
        n_atoms = self.data['n_atoms'][idx]
        edge_indices = self.data['edge_indices'][:, e_start: e_end]

        mol_graph = dgl.graph((edge_indices[0], edge_indices[1]), num_nodes=n_atoms)
        mol_graph.ndata['feat'] = self.data['all_atom_features'][start: start + n_atoms]
        mol_graph.edata['feat'] = self.data['all_edge_features'][e_start: e_end]
        return mol_graph
    def __len__(self):
        return len(self.data['atom_slices']) - 1

    def __getitem__(self, idx):
        data = []
        g = self.mol_2_graph(idx)
        data.append(g)
        data.append(self.data['morgan_fingers'][idx].to(self.device))
        data.append(self.data['protein_feats'][idx].to(self.device))
        data.append(self.data['protein_feats_dist'][idx].to(self.device))
        data.append(self.data['labels'][idx].to(self.device))
        return tuple(data)
