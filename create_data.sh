#!/bin/bash
echo 'enter GPU' 
read gpu
# data name, data(protein_sequence following esm dict), output_file, distance metric pdb file, esm prediction folder
CUDA_VISIBLE_DEVICES=$gpu python create_train_cuscpidata_ecfp.py 'test_git' '/ssd0/quang/dti2d/dataset/cluster_validation_dif/git_test' '/ssd0/quang/dti2d/dataset/cluster_validation_dif/git_test_processed' '/ssd1/quang/dti2d/dataset/esm1davis_data_possitioninfor1/davis_dis.pt' '/ssd1/quang/dti2d/dataset/esm/esm1davis_data' 


