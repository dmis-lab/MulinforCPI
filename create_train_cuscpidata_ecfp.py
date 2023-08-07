import os
from datasets.cus_dataset_cpi_processed3 import CusDatasetCPI
import biotite.structure as struc
import biotite.structure.io as strucio
# turn on for debugging C code like Segmentation Faults
import faulthandler
import time
import torch

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.set_num_threads(1)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"
import sys
def datamaker(dataname, esm_fold_split, processed_fold_name, distance_matrix_pdb, protein_fold_folder):
    
    if dataname == 'davis':
        tasks = ['novel_prot','novel_pair','novel_comp']
    else:
        tasks = ['novel_pair']

    for task in tasks:
        files = os.listdir(os.path.join(esm_fold_split,task))
        for file in files:
            print(os.path.join(os.path.join(esm_fold_split,task),file))
            data = CusDatasetCPI(smiles_txt_path = os.path.join(os.path.join(esm_fold_split,task),file),
             dataname = dataname, 
             processed_fold_name = processed_fold_name, 
             protein_pdb_fold = protein_fold_folder, 
             res_dis = distance_matrix_pdb)

if __name__ == '__main__':

    datamaker(dataname = str(sys.argv[1]),
              esm_fold_split  = str(sys.argv[2]),
              processed_fold_name  = str(sys.argv[3]),
              distance_matrix_pdb  = str(sys.argv[4]),
              protein_fold_folder  = str(sys.argv[5]))

