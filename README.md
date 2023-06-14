# 3DinforCPI

A Pytorch Implementation of paper:

**3DinforCPI:Enhancing Compound-Protein Interaction Prediction Efficiency through Novel Perspectives on 3D Information Integration**

Our reposistory uses 3DInformax from https://github.com/HannesStark/3DInfomax as a backbone for pretraining PNA for compound information extraction and ESM_Fold from https://github.com/facebookresearch/esm for predicting protein fold.

# 4.**To train YOUR model:**
Your data should be in the format csv, and the column names are: 'smiles','sequence','label'.
1. Generate the 3D fold of protein from the dataset.
~~~
python generate_protein_fold.py #datafolder
~~~
2. Calcualte the Carbon Alpha Carbon distance.
~~~
python positional_ebedding_dist.py
~~~
3. Align the training dataset following the output of ESM.
~~~
python match_protein_index_esm.py
~~~
