import os
import torch
import biotite.structure as struc
import biotite.structure.io as strucio
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import sys


class makedisdataset():
    def __init__(self, input_folder, output_folder, data, protein_length = 500):
        self.input_folder = input_folder
        self.output_folder = output_folder
        if not os.path.exists(output_folder): os.makedirs(output_folder)
        self.protein_features_dist = []
        self.data = data

    def make_distance_matrix(self):
        files = os.listdir(self.input_folder)
        pdb_files = [file for file in files if file.endswith('.pdb')]
        for index, file in enumerate(tqdm(pdb_files)):
            struct = strucio.load_structure(os.path.join(self.input_folder,'{}_result.pdb'.format(index)))
            CA_coords = [ atom.coord for atom in struct if atom.atom_name == 'CA' ]

            dist_matrix = np.zeros((protein_length, protein_length), np.float64)
            for row, res1 in enumerate(CA_coords):
                for col, res2 in enumerate(CA_coords):
                    try:
                        dist_matrix[row,col] = np.sqrt(np.sum((res1-res2)**2, axis=0))
                    except:
                        pass
            self.protein_features_dist.append(dist_matrix)
        torch.save(torch.from_numpy(np.array(self.protein_features_dist)),
         os.path.join(self.output_folder,'{}_dis.pt').format(self.data))

def main(input_folder, output_folder, data_name = data_name):
    makedataset = makedisdataset(input_folder = input_folder,\
        output_folder = output_folder, data = data_name)
    makedataset.make_distance_matrix()
if __name__ == '__main__':
    input_folder = str(sys.argv[1])
    output_folder = str(sys.argv[2])
    data_name = str(sys.argv[3])
    main(input_folder = input_folder, \
        output_folder= output_folder,\
        data_name = data_name)