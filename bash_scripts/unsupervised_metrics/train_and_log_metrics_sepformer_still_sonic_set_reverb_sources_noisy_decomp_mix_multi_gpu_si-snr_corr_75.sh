#!/bin/bash 

# 

#SBATCH --job-name=train-sepformer-log-metrics-still-sonic-set-decomp_mix_rev_sources_corr_loss_75 #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --gpus=4  #number of gpus requested
#SBATCH --partition=gpu
#SBATCH --error=/export/fs05/afrumme1/git/speechbrain/logs/still_sonic_set_v1_reverb_sources_noisy_decomp_mix_log_metrics_multi_gpu_corr_loss_75/extract.err
#SBATCH --output=/export/fs05/afrumme1/git/speechbrain/logs/still_sonic_set_v1_reverb_sources_noisy_decomp_mix_log_metrics_multi_gpu_corr_loss_75/extract.out
#SBATCH --mail-user=afrumme1@jh.edu  #email for reporting
#SBATCH --mail-type=END,FAIL,BEGIN

echo "Script started at: $(date)"
echo "train sepformer still_sonic_set_v1_reverb_sources_noisy_decomp_mix dataset using 4 GPUS"
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
module purge 
module load conda 

source activate /home/afrumme1/miniconda3/envs/speechbrain
export PYTHONPATH=/export/fs05/afrumme1/git/speechbrain:$PYTHONPATH

wandb login 16f04d1c9cc464e10afc56b3c128d3c7c9f7ef13
/home/afrumme1/miniconda3/envs/speechbrain/bin/torchrun --nproc_per_node=4 /export/fs05/afrumme1/git/speechbrain/recipes/WSJ0Mix/separation/train_with_metrics_log_correlation_loss.py /export/fs05/afrumme1/git/speechbrain/recipes/WSJ0Mix/separation/hparams_copy/unsupervised_losses/sepformer_log_metrics-still_sonic_set_decomp_mixtire_rev_sources_multi_gpu_unseparation_loss_75.yaml --data_folder /export/fs05/afrumme1/sepformer_training/still_sonic_set_v1_reverb_sources_noisy_decomp_mix


# Capture exit status
if [ $? -eq 0 ]; then
    echo "train sepformer on noisy-still-sonic-set mixture dataset using 4 GPUS completed successfully!"
else
    echo "Error: train sepformer on noisy-still-sonic-set mixture dataset using 1 GPUS - Failed." >&2
fi

echo "Script finished at: $(date)"