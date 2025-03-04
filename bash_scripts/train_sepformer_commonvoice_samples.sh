#!/bin/bash 

# 

#SBATCH --job-name=sep-com-voice #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --gpus=4  #number of gpus requested
#SBATCH --partition=gpu
#SBATCH --error=logs/extract.err
#SBATCH --output=logs/extract.out
#SBATCH --mail-user=afrumme1@jh.edu  #email for reporting

echo "train sepformer on common voice large mixture dataset"
module purge 
module load conda 

source activate /home/afrumme1/miniconda3/envs/speechbrain

/home/afrumme1/miniconda3/envs/speechbrain/bin/python /home/afrumme1/speechbrain/recipes/WSJ0Mix/separation/train.py /home/afrumme1/speechbrain/recipes/WSJ0Mix/separation/hparams/sepfromer-common-voice.yaml --data_folder /export/fs05/afrumme1/commonvoice_experiment