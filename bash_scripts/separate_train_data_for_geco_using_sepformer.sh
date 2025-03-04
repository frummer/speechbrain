#!/bin/bash 

# 

#SBATCH --job-name=sep-com-voice-train-data-for-geco-using-1-gpu #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --gpus=1  #number of gpus requested
#SBATCH --partition=gpu-a100
#SBATCH --error=logs/extract.err
#SBATCH --output=logs/extract.out
#SBATCH --mail-user=afrumme1@jh.edu  #email for reporting
#SBATCH --account=a100acct 

echo "separate train data for geco using sepformer with 1 GPU"
module purge 
module load conda 

source activate /home/afrumme1/miniconda3/envs/speechbrain

/home/afrumme1/miniconda3/envs/speechbrain/bin/python /home/afrumme1/speechbrain/recipes/WSJ0Mix/separation/train.py /home/afrumme1/speechbrain/recipes/WSJ0Mix/separation/hparams/separate_commonvoice_using_sepformer_train_set.yaml --data_folder /export/fs05/afrumme1/commonvoice_experiment --test_only