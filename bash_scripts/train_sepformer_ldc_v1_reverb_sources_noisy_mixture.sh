#!/bin/bash 

# 

#SBATCH --job-name=sep-ldc-noisemix_rev_source #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --gpus=2 #number of gpus requested
#SBATCH --partition=gpu-a100
#SBATCH --account=a100acct 
#SBATCH --error=/home/afrumme1/speechbrain/logs/sldc_v1_reverb_sources_noisy_mixture/extract.err
#SBATCH --output=/home/afrumme1/speechbrain/logs/sldc_v1_reverb_sources_noisy_mixture/extract.out
#SBATCH --mail-user=afrumme1@jh.edu  #email for reporting
#SBATCH --mail-user=afrumme1@jh.edu  #email for reporting
#SBATCH --mail-type=END,FAIL,BEGIN

echo "train sepformer on ldc_v1_reverb_sources_noisy_mixture dataset using 4 GPUS"
echo "Script started at: $(date)"

module purge 
module load conda 
nvidia-smi
export CUDA_VISIBLE_DEVICES=0,1
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
source activate /home/afrumme1/miniconda3/envs/speechbrain

/home/afrumme1/miniconda3/envs/speechbrain/bin/torchrun --nproc_per_node=2 /home/afrumme1/speechbrain/recipes/WSJ0Mix/separation/train.py /home/afrumme1/speechbrain/recipes/WSJ0Mix/separation/hparams/sepformer-ldc_v1_reverb_sources_noisy_mixture.yaml --data_folder /export/fs05/afrumme1/sepformer_training/ldc_v1_reverb_sources_noisy_mixture


# Capture exit status
if [ $? -eq 0 ]; then
    echo "train sepformer on ldc_v1_reverb_sources_noisy_mixture dataset using 4 GPUS completed successfully!"
else
    echo "Error: train sepformer on ldc_v1_reverb_sources_noisy_mixture dataset using 4 GPUS - Failed." >&2
fi

echo "Script finished at: $(date)"