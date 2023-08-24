# 3DinforCPI

A PyTorch Implementation of Paper:

**3DinforCPI: Enhancing Compound-Protein Interaction Prediction Efficiency through Novel Perspectives on multi-Level Information Integration**

Our repository uses 3DInformax from https://github.com/HannesStark/3DInfomax as a backbone for pretraining PNA for compound information extraction and ESM_Fold from https://github.com/facebookresearch/esm for predicting protein fold.

Forecasting the interaction between compounds and proteins is crucial for discovering new drugs. However, several commonly used computational techniques assume that the molecules or proteins in the test set have already been observed during the training phase. This assumption might not remain valid in real-life scenarios, where the test and training data could exhibit varying distributions and stem from distinct domains. Moreover, previous studies have not utilized three-dimensional (3D) information on compounds and proteins to predict binding affinity. Here, we propose a two-step deep learning strategy named 3DinforCPI that incorporates transfer learning techniques with multi-level resolution features to overcome these limitations. Our approach leverages 3D information from both proteins and compounds and acquires a profound understanding of the atomic-level features of proteins. We applied our approach to six datasets: Davis, Metz, KIBA, CASF-2016, DUD-E, and BindingDB, to evaluate the effectiveness of our approach.
<img src="images/architecture2.PNG" alt="Image" width="1000" >


## **Clustering the dataset:**
In our experiment we use cross, we used the cross-cluster validation technique. Leave one out for testing while the validation set is randomly taken from training set with the ratio 20/80.<br />
  `data_file: The file contains dataset (Davis,KIBA,metz)`<br />
  `output_folder: The folder contains five clusters`<br />
  ~~~
  python prepare_cluster_data_2023 #data_file #output_folder
  ~~~
<img src="images/PCA_clustering.png" alt="Image" width="600" >



## **To train 3DinforCPI model:**

Your data should be in the format .csv, and the column names are: 'smiles', 'sequence', 'label'.
1. Generate the 3D fold of protein from the dataset.<br />
`data_folder: Folder of dataset`<br />
  ~~~
  python generate_protein_fold.py #data_folder
  ~~~
<img src="images/ESM_output.png" alt="Image" width="600" >
The output of ESM fold. The 3D fold contains 1) The Card, 2) Atom Number, 3) Atom Type, 4) Three-Letter Amino Acid Code, 5) Chain ID, 6) Residue Number, 7) Atom Coordinates, 8) Atom Occupancy, 9) Atomic Displacement Parameter, 10) Element.

2. Calculate the Alpha Carbon distances.<br />
`input_folder: Output folder from ESM prediction.(Output of step 1.)`<br />
`output_folder: Folder of processed file`<br />
`data_name: Name of dataset`<br />
  ~~~
  python generate_distance_map.py #input_folder #output_folder #data_name
  ~~~

  3. Align the training dataset following the output of ESM fold. (FOR data-making purposes)<br />
`data_folder: Folder of Training dataset` <br />
`output_folder: Folder of processed file` <br />
`prot_dict: _data.csvdic.csv file in output folder of ESM prediction.`<br />
  ~~~
  python match_protein_index_esm.py #data_folder #output_folder #prot_dict
  ~~~

4. Generate the pickle .pt file for training <br />
`data name: Folder of Training dataset` <br />
`data_path: Aligned dataset (Output of step 3.)` <br />
`output_folder: Folder of Training dataset` <br />
`distance metric pt file: Alpha Carbon distances. ( Output of step 2.)` <br />
`esm prediction folder: Output folder from ESM prediction. (Output of step 1.)` <br />

  ~~~
  python create_train_cuscpidata_ecfp.py #data_name #data_path #output_folder #distance_metric_pdb_file #esm_prediction_folder 
  ~~~
5. Train the model <br />
 Change the `data_path: the processed data folder in .pt format ( Output of step 4.)` in `best_configs/tune_cus_cpi.yml`  <br />
  ~~~
  python train_cuscpi.py --config best_configs/tune_cus_cpi.yml
  ~~~
## **Dataset:**
The related links are as follows: <br />

KIBA, Davis:https://github.com/kexinhuang12345/DeepPurpose <br />
Metz: https://github.com/sirimullalab/KinasepKipred <br />
BindingDB: https://github.com/IBM/InterpretableDTIP <br />
DUD-E Diverse: http://dude.docking.org/subsets/diverse <br />
QMugs: https://libdrive.ethz.ch/index.php/s/X5vOBNSITAG5vzM <br />
CASF-2016: http://www.pdbbind.org.cn/casf.php <br />


## **To take inference:**
  Change the `test_data_path` and `checkpoint` in best_configs/inference_cpi.yml to take the inference (with `test_data_path` in made following step 1-2-3) <br />
  ~~~
  python inferencecpi.py --config=best_configs/inference_cpi.yml
  ~~~


## License
[MIT](https://choosealicense.com/licenses/mit/)
