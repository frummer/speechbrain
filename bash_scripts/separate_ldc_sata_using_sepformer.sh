#!/bin/bash 

# 

#SBATCH --job-name=sep-com-voice-test-data-for-geco-using-1-gpu #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --gpus=1  #number of gpus requested
#SBATCH --partition=gpu-a100
#SBATCH --error=/home/afrumme1/speechbrain/logs/inference_sepformer/extract.err
#SBATCH --output=/home/afrumme1/speechbrain/logs/inference_sepformer/extract.out
#SBATCH --mail-user=afrumme1@jh.edu  #email for reporting
#SBATCH --account=a100acct 


echo "separate ldc test data for using sepformer with 1 GPU"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi
module purge 
module load conda 

source activate /home/afrumme1/miniconda3/envs/speechbrain

/home/afrumme1/miniconda3/envs/speechbrain/bin/python /home/afrumme1/speechbrain/recipes/WSJ0Mix/separation/train.py /home/afrumme1/speechbrain/recipes/WSJ0Mix/separation/hparams/separate_ldc_using_sepformer_test_set.yaml --data_folder /export/fs05/afrumme1/sepformer_training/ldc_v1_reverb_sources_noisy_mixture --test_only