# 3DinforCPI

A Pytorch Implementation of paper:

**3DinforCPI:Enhancing Compound-Protein Interaction Prediction Efficiency through Novel Perspectives on multi-Level Information Integration**

Our reposistory uses 3DInformax from https://github.com/HannesStark/3DInfomax as a backbone for pretraining PNA for compound information extraction and ESM_Fold from https://github.com/facebookresearch/esm for predicting protein fold.

## **To train YOUR model:**

Your data should be in the format csv, and the column names are: 'smiles','sequence','label'.
1. Generate the 3D fold of protein from the dataset.<br />
`data_folder: Folder of dataset`<br />
  ~~~
  python generate_protein_fold.py #data folder
  ~~~
2. Calcualte the Carbon Alpha Carbon distance.<br />
`input_folder: Output folder from ESM prediction.`<br />
`output_folder: Folder of processed file`<br />
`data_name: Name of dataset`<br />
  ~~~
  python generate_distance_map.py #input_folder #output_folder #data_name
  ~~~

  3. Align the training dataset following the output of ESM. (FOR data making purpose)<br />
`data_folder: Folder of Training dataset` <br />
`output_folder: Folder of processed file` <br />
`prot_dict: _data.csvdic.csv file in output folder of ESM prediction.`<br />

  ~~~
  python match_protein_index_esm.py #data_folder #output_folder #prot_dict
  ~~~





## License
[MIT](https://choosealicense.com/licenses/mit/)
