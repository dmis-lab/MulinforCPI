import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import sys


# print(sys.argv[1])
# import pdb
# pdb.set_trace()
# data_name = davis, kiba, metz

def main(data_folder, out_folder, prot_dict, data_name):

    # data_name = 'binddingdb'
    # data_folder = '/ssd1/quang/dti2d/dataset/cluster_binddingdb'
    # out_folder = '/ssd1/quang/dti2d/dataset/split_{}_esm1'.format(data_name)
    # prot_dict = '/ssd1/quang/dti2d/dataset/esm/esm1{}_data/{}_data.csvdic.csv'.format(data_name,data_name)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    if data_name == 'davis':
        tasks = ['novel_pair','novel_comp','novel_prot']
    else:
        tasks = ['novel_pair']

    for task in tasks:
        cur_dir = os.path.join(data_folder,task)
        files = os.listdir(os.path.join(data_folder,task))
        if not os.path.exists(os.path.join(out_folder,task)):
            os.makedirs(os.path.join(out_folder,task))
        for file in files:
            dataset = os.path.join(cur_dir,file)
            df_prot_dict = pd.read_csv(prot_dict)
            df_data = pd.read_csv(dataset)
            df_out = []
            for i in tqdm(range(len(df_data))):
                row = df_data.loc[i]
                for j in range(len(df_prot_dict)):
                    if df_data.loc[i]['sequence'] == df_prot_dict.loc[j]['sequence']:
                        row['sequence'] = df_prot_dict.loc[j]['index']
                        df_out.append(row)
            df = pd.DataFrame(df_out)
            df.to_csv(os.path.join(os.path.join(out_folder,task),file))

if __name__ == '__main__':
    data_folder = str(sys.argv[1])
    output_folder = str(sys.argv[2])
    prot_dict = str(sys.argv[3])
    data_name = str(sys.argv[4])
    main(data_folder, out_folder, prot_dict, data_name)
