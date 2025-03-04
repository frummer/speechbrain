#!/bin/bash 

# 

#SBATCH --job-name=ldv-v1-sep-train-multi-gpu #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --gpus=1  #number of gpus requested
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct 
#SBATCH --error=/home/afrumme1/speechbrain/logs/ldc_v1_sepformer_training/extract.err
#SBATCH --output=/home/afrumme1/speechbrain/logs/ldc_v1_sepformer_training/extract.out
#SBATCH --mail-user=afrumme1@jh.edu  #email for reporting
#SBATCH --mail-type=END,FAIL,BEGIN
echo "train sepformer on ldcv1 mixture dataset using 1 GPUS"
echo "Script started at: $(date)"
module purge 
module load conda 
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

source activate /home/afrumme1/miniconda3/envs/speechbrain
export CUDA_VISIBLE_DEVICES=0
/home/afrumme1/miniconda3/envs/speechbrain/bin/python /home/afrumme1/speechbrain/recipes/WSJ0Mix/separation/train.py /home/afrumme1/speechbrain/recipes/WSJ0Mix/separation/hparams/sepformer-ldc-training.yaml --data_folder /export/fs05/afrumme1/sepformer_training/ldc_v1_experiment

# Capture exit status
if [ $? -eq 0 ]; then
    echo "train sepformer on ldcv1 mixture dataset using 1 GPUS completed successfully!"
else
    echo "Error: train sepformer on ldcv1 mixture dataset using 1 GPUS." >&2
fi

echo "Script finished at: $(date)"