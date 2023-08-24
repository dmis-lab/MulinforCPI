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

# res_names = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'] 
# atom_names = ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2', 'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1', 'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'SD', 'SG']
class CusDatasetCPI(Dataset):

    def __init__(self, smiles_txt_path, device='cuda:0', transform=None, **kwargs):
        self.protein_pdb_fold = "/ssd1/quang/dti2d/dataset/esm/esm1davis_data"
        self.res_dis = torch.load("/ssd1/quang/dti2d/dataset/esm1davis_data_possitioninfor1/davis_dis.pt")
        self.res_names = self.take_unique(self.protein_pdb_fold,'res_name')
        self.atom_names = self.take_unique(self.protein_pdb_fold,'atom_name')

        self.smiles_list = list(pd.read_csv(smiles_txt_path)['smiles'])
        self.sequenceindex_list = list(pd.read_csv(smiles_txt_path)['sequence'])
        self.labels = torch.tensor(list(pd.read_csv(smiles_txt_path)['label']))

        self.device = device 
##making feature for graph
        self.n_atoms, self.atom_slices, self.edge_slices, self.edge_indices,\
        self.all_atom_features, self.all_edge_features = self.mol_featurization()
##making 1hot feature for protein
        self.protein_feats , self.protein_feats_dist = self.protein_featurization()

    def protein_featurization(self):
        protein_features = []
        protein_features_dist = []
        
        for index, protein_index in enumerate(tqdm(self.sequenceindex_list)):
            struct = strucio.load_structure(os.path.join(self.protein_pdb_fold,\
                str(protein_index)+'_result.pdb'))
            #residue
            resnames_tensor = torch.tensor([self.res_names.index(i) for i in struct.res_name])
            protein_feature0 = F.one_hot(resnames_tensor, num_classes = len(self.res_names))
            #atom
            atomnames_tensor = torch.tensor([self.atom_names.index(i) for i in struct.atom_name])
            protein_feature1= F.one_hot(atomnames_tensor, num_classes = len(self.atom_names))
            #element
            protein_feature2 = torch.tensor([atom_to_feature_vector(Chem.MolFromSmiles(smiles).GetAtoms()[0]) for smiles in struct.element])
            #distance matrix
            # for atom in struct:
            #     if atom.atom_name == 'CA':

            # CA_coords = [atom.coord for atom in struct if atom.atom_name == 'CA' ]
            # # import pdb
            # # pdb.set_trace()

            # dist_matrix = np.zeros((500, 500), np.float)
            # for row, res1 in enumerate(CA_coords):
            #     for col, res2 in enumerate(CA_coords):
            #         try:
            #             dist_matrix[row,col] = np.sqrt(np.sum((res1-res2)**2, axis=0))
            #         except:
            #             pass
            # print(dist_matrix)
            protein_features_dist.append(self.res_dis[protein_index].numpy())
            # import pdb
            # pdb.set_trace()
            #protein
            protein_feature = torch.cat((protein_feature0, protein_feature1, protein_feature2),axis = 1)

            dummy_matric = torch.zeros(4000,protein_feature.size(1))
            protein_feature = torch.cat((protein_feature, dummy_matric),axis = 0)
            protein_features.append(protein_feature[:4000].numpy())
            # torch.vstack()
            # import pdb
            # pdb.set_trace()
        # return torch.cat(protein_features, dim=0)
        return torch.from_numpy(np.array(protein_features)),\
        torch.from_numpy(np.array(protein_features_dist))

    def mol_featurization(self):
        atom_slices = [0]
        edge_slices = [0]
        all_atom_features = []
        all_edge_features = []
        edge_indices = []  # edges of each molecule in coo format
        total_atoms = 0
        total_edges = 0
        n_atoms_list = []
        for mol_idx, smiles in enumerate(tqdm(self.smiles_list)):
            # get the molecule using the smiles representation from the csv file
            mol = Chem.MolFromSmiles(smiles)
            # add hydrogen bonds to molecule because they are not in the smiles representation
            mol = Chem.AddHs(mol)
            n_atoms = mol.GetNumAtoms()

            atom_features_list = []
            for atom in mol.GetAtoms():
                atom_features_list.append(atom_to_feature_vector(atom))
            all_atom_features.append(torch.tensor(atom_features_list, dtype=torch.long))

            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_feature = bond_to_feature_vector(bond)
                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)
            # Graph connectivity in COO format with shape [2, num_edges]
            edge_index = torch.tensor(edges_list, dtype=torch.long).T
            edge_features = torch.tensor(edge_features_list, dtype=torch.long)

            edge_indices.append(edge_index)
            all_edge_features.append(edge_features)
            total_edges += len(edges_list)
            total_atoms += n_atoms
            edge_slices.append(total_edges)
            atom_slices.append(total_atoms)
            n_atoms_list.append(n_atoms)

        n_atoms = torch.tensor(n_atoms_list)
        atom_slices = torch.tensor(atom_slices, dtype=torch.long)
        edge_slices = torch.tensor(edge_slices, dtype=torch.long)
        edge_indices = torch.cat(edge_indices, dim=1)
        all_atom_features = torch.cat(all_atom_features, dim=0)
        all_edge_features = torch.cat(all_edge_features, dim=0)
        
        return n_atoms, atom_slices, edge_slices, edge_indices,\
        all_atom_features, all_edge_features

    def mol_2_graph(self, idx):
        e_start = self.edge_slices[idx]
        e_end = self.edge_slices[idx + 1]
        start = self.atom_slices[idx]
        n_atoms = self.n_atoms[idx]
        edge_indices = self.edge_indices[:, e_start: e_end]
        mol_graph = dgl.graph((edge_indices[0], edge_indices[1]), num_nodes=n_atoms)
        mol_graph.ndata['feat'] = self.all_atom_features[start: start + n_atoms]
        mol_graph.edata['feat'] = self.all_edge_features[e_start: e_end]
        return mol_graph

    def take_unique(self, protein_pdb_fold, type_feature):
        # protein_pdb_fold = "/hdd1/quang_backups/dti_2d/dataset/esm/esm1davis_data"
        list_files = os.listdir(protein_pdb_fold)
        unique_list = []

        if type_feature == 'atom_name':
            for index, protein_file in enumerate(tqdm(list_files)):
                if protein_file.endswith('.pdb'):
                    struct = strucio.load_structure(os.path.join(protein_pdb_fold,\
                        protein_file))
                    unique_list.extend(list(set(struct.atom_name)))
                else:
                    pass
            return sorted(set(unique_list))
        if type_feature == 'res_name':
            for index, protein_file in enumerate(tqdm(list_files)):
                if protein_file.endswith('.pdb'):
                    struct = strucio.load_structure(os.path.join(protein_pdb_fold,\
                        protein_file))
                    unique_list.extend(list(set(struct.res_name)))
                else:
                    pass
            return sorted(set(unique_list))
    def __len__(self):
        return len(self.atom_slices) - 1

    def __getitem__(self, idx):
        data = []
        e_start = self.edge_slices[idx]
        e_end = self.edge_slices[idx + 1]
        start = self.atom_slices[idx]
        n_atoms = self.n_atoms[idx]
        edge_indices = self.edge_indices[:, e_start: e_end]
        g = self.mol_2_graph(idx)
        data.append(g)
        data.append(self.protein_feats[idx].to(self.device))
        data.append(self.protein_feats_dist[idx].to(self.device))
        data.append(self.labels[idx].to(self.device))
        return tuple(data)
