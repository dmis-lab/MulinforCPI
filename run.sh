#!/bin/bash
echo 'enter GPU' 
read gpu
echo "Enter a number:"
read num

# Use a for loop to count from 1 to the user input
for (( i=1; i<=$num; i++ ))
do
	CUDA_VISIBLE_DEVICES=$gpu python train_cuscpi.py --config best_configs/tune_cus_cpi1.yml
done
# Exit the script
exit 0
