#!/bin/bash 

# 

#SBATCH --job-name=separation-train_libriheavyMix_2spk_multi_gpu #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --gpus=4  #number of gpus requested
#SBATCH --partition=gpu
#SBATCH --error=logs_multi_gpu/extract.err
#SBATCH --output=logs_multi_gpu/extract.out
#SBATCH --mail-user=afrumme1@jh.edu  #email for reporting

echo "Startig to organize libriheavymix 2 speakers for training"
module purge 
module load conda 

source activate /home/afrumme1/miniconda3/envs/speechbrain

/home/afrumme1/miniconda3/envs/speechbrain/bin/python /home/afrumme1/speechbrain/recipes/WSJ0Mix/separation/train.py /home/afrumme1/speechbrain/recipes/WSJ0Mix/separation/hparams/sepfromer-common-voice.yaml
