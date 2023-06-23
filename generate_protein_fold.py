import torch
import esm
from tqdm import tqdm
import numpy as np
import pdb
import pandas as pd
import os
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import sys

device = torch.device("cuda:1")
model = esm.pretrained.esmfold_v1().cuda()
model = model.eval()
model.to(device)
# Optionally, uncomment to set a chunk size for axial attention. This can help reduce memory.
# Lower sizes will have lower memory requirements at the cost of increased speed.

def main(data_path):
    files = os.listdir(data_path)
    for file in files:
        if file.endswith('.csv'):
            re_fold = os.path.join(data_folder,'esm1' + file[:-4])
            if not os.path.isdir(re_fold): os.makedirs(re_fold)

            data1 = pd.read_csv(os.path.join(data_folder,file))
            data = data1
            uni_sequences = sorted(list(set(data['sequence'])), key = len)

            for index, seq in enumerate(tqdm(list(data['sequence']))):
                data['sequence'].loc[index] = uni_sequences.index(seq)
                
            list_match = []

            for index, seq in enumerate(tqdm(uni_sequences)):
                list_match.append([seq, index])
            list_mach_df = pd.DataFrame(list_match, columns = ['sequence', 'index'])
            list_mach_df.to_csv(os.path.join(re_fold, file + 'dic.csv'))
            data.to_csv(os.path.join(re_fold,'esm'+file))

            for index, sequence in enumerate(tqdm(uni_sequences)):
                data_last = {}
                with torch.no_grad():
                    output, output_distogram_logits, output_ptm_logits, output_lm_logits = model.infer_pdb(sequence[:500])
                    # data_last.update({'distogram_logits':output_distogram_logits,
                    #              'ptm_logits':output_ptm_logits, 
                    #              'lm_logits':output_lm_logits})
                    # torch.save(data_last, os.path.join(re_fold,"{}_result_last.pt".format(index)))
                with open(os.path.join(re_fold,"{}_result.pdb".format(index)), "w") as f:
                    f.write(output[0])

if __name__ == '__main__':
    data_path = str(sys.argv[1])
    main(data_path=data_path)
