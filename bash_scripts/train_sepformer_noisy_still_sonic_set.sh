#!/bin/bash 

# 

#SBATCH --job-name=noisy-still-sset-sep-train-multi-gpu #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --gpus=1  #number of gpus requested
#SBATCH --partition=gpu
#SBATCH --error=/home/afrumme1/speechbrain/logs/noisy_still_sonic_set/extract.err
#SBATCH --output=/home/afrumme1/speechbrain/logs/noisy_still_sonic_set/extract.out
#SBATCH --mail-user=afrumme1@jh.edu  #email for reporting
#SBATCH --mail-type=END,FAIL,BEGIN
echo "train sepformer on noisy-still-sonic-set mixture dataset using 4 GPUS"
echo "Script started at: $(date)"
module purge 
module load conda 
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

source activate /home/afrumme1/miniconda3/envs/speechbrain
/home/afrumme1/miniconda3/envs/speechbrain/bin/python /home/afrumme1/speechbrain/recipes/WSJ0Mix/separation/train.py /home/afrumme1/speechbrain/recipes/WSJ0Mix/separation/hparams/sepformer-noisy_still_sonic_set_training.yaml --data_folder /export/fs05/afrumme1/sepformer_training/still_sonic_set_v1_experiment

# Capture exit status
if [ $? -eq 0 ]; then
    echo "train sepformer on noisy-still-sonic mixture dataset using 1 GPU completed successfully!"
else
    echo "Error: train sepformer on noisy-still-sonic mixture dataset using 1 GPU." >&2
fi

echo "Script finished at: $(date)"