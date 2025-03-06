#!/bin/bash 

# 

#SBATCH --job-name=sep-still-sonic-set-test #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --gpus=1  #number of gpus requested
#SBATCH --partition=gpu
#SBATCH --error=/export/fs05/afrumme1/git/speechbrain/logs/separate_still_sonic_set_v1_reverb_sources_noisy_decomp_mix/extract.err
#SBATCH --output=/export/fs05/afrumme1/git/speechbrain/logs/separate_still_sonic_set_v1_reverb_sources_noisy_decomp_mix/extract.out
#SBATCH --mail-user=afrumme1@jh.edu  #email for reporting
#SBATCH --mail-user=afrumme1@jh.edu  #email for reporting
#SBATCH --mail-type=END,FAIL,BEGIN

echo "train sepformer on noisy-still-sonic-set mixture dataset using 4 GPUS"
echo "Script started at: $(date)"
echo "train sepformer still_sonic_set_v1_reverb_sources_noisy_decomp_mix dataset using 4 GPUS"

module purge 
module load conda 
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
source activate /home/afrumme1/miniconda3/envs/speechbrain

/home/afrumme1/miniconda3/envs/speechbrain/bin/python /export/fs05/afrumme1/git/speechbrain/recipes/WSJ0Mix/separation/train.py /export/fs05/afrumme1/git/speechbrain/recipes/WSJ0Mix/separation/hparams_copy/inference-test-sepformer-still_sonic_set_decomp_mixtire_rev_sources.yaml --data_folder /export/fs05/afrumme1/sepformer_training/still_sonic_set_v1_reverb_sources_noisy_decomp_mix --test_only


# Capture exit status
if [ $? -eq 0 ]; then
    echo "train sepformer on noisy-still-sonic-set mixture dataset using 4 GPUS completed successfully!"
else
    echo "Error: train sepformer on noisy-still-sonic-set mixture dataset using 4 GPUS - Faiels." >&2
fi

echo "Script finished at: $(date)"