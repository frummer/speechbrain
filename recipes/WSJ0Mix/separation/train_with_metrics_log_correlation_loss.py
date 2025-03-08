#!/usr/bin/env/python3
"""Recipe for training a neural speech separation system on the wsjmix
dataset. The system employs an encoder, a decoder, and a masking network.

To run this recipe, do the following:
> python train.py hparams/sepformer.yaml
> python train.py hparams/dualpath_rnn.yaml
> python train.py hparams/convtasnet.yaml

The experiment file is flexible enough to support different neural
networks. By properly changing the parameter files, you can try
different architectures. The script supports both wsj2mix and
wsj3mix.


Authors
 * Cem Subakan 2020
 * Mirco Ravanelli 2020
 * Samuele Cornell 2020
 * Mirko Bronzi 2020
 * Jianyuan Zhong 2020
"""

import csv
import os
import sys

import numpy as np
from leakage_utils import compute_leakage_for_pair
from ellipsis_utils import compute_ellipsis_metric
import torch
import torch.nn.functional as F
import torchaudio
from hyperpyyaml import load_hyperpyyaml
import concurrent.futures
from tqdm import tqdm
import torch.utils.data as data
if os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) != "0":
    os.environ["WANDB_MODE"] = "disabled"
import wandb
import speechbrain as sb
import speechbrain.nnet.schedulers as schedulers
from speechbrain.core import AMPConfig
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.logger import get_logger
from torch.utils.data import Subset


# Define training procedure
class Separation(sb.Brain):
    def evaluate(
        self,
        test_set,
        max_key=None,
        min_key=None,
        progressbar=None,
        test_loader_kwargs={},
    ):
        """
        Evaluate brain performance on a test set, and compute unseparation
        for the entire test_set by calling `on_stage_end(Stage.TEST, dataset=...)`.
        """
        # By default, load the best checkpoint:
        self.on_evaluate_start(max_key=max_key, min_key=min_key)

        if progressbar is None:
            progressbar = not self.noprogressbar
        enable = progressbar and sb.utils.distributed.if_main_process()

        # Turn a Dataset into a DataLoader if needed:
        if not (
            isinstance(test_set, sb.dataio.dataloader.SaveableDataLoader)
            or isinstance(test_set, sb.dataio.dataloader.LoopedLoader)
        ):
            # Prevent the test dataloader from being saved as a checkpoint:
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(test_set, stage=sb.Stage.TEST, **test_loader_kwargs)

        # Standard SB calls:
        self.on_stage_start(sb.Stage.TEST, epoch=None)
        self.modules.eval()

        avg_test_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(test_set, disable=not enable, colour=self.tqdm_barcolor["test"]):
                self.step += 1
                loss = self.evaluate_batch(batch, stage=sb.Stage.TEST)
                avg_test_loss = self.update_average(loss, avg_test_loss)

                if self.debug and self.step == self.debug_batches:
                    break
                
        self.on_stage_end(sb.Stage.TEST, avg_test_loss, epoch=None)
        self.step = 0    

    def _fit_train(self, train_set, epoch, enable):
         # 1) Call the standard (parent) training loop
        super()._fit_train(train_set, epoch, enable)
        """Custom validation loop that passes dataset to on_stage_end()."""
        # Create a subset of the underlying dataset with the first 5 indices.
        subset_dataset = Subset(train_set.dataset, list(range(2000)))

        # Re-create a SaveableDataLoader for the subset using the same parameters as train_set.
        subset_dataloader = sb.dataio.dataloader.SaveableDataLoader(
            subset_dataset,
            batch_size=train_set.batch_size,     # use the same batch size
            shuffle=False,                       # disable shuffling for logging, if needed
            collate_fn=train_set.collate_fn,     # use the same collate function
            # ... include any other parameters from train_set that are needed
        )
        if sb.utils.distributed.if_main_process():
            self.log_unsupervies_metrics(stage=sb.Stage.TRAIN, epoch=epoch, dataset=subset_dataloader)
        
    def _fit_valid(self, valid_set, epoch, enable):
         # 1) Call the standard (parent) training loop
        super()._fit_valid(valid_set, epoch, enable)
        """Custom validation loop that passes dataset to on_stage_end()."""
        if sb.utils.distributed.if_main_process():
            self.log_unsupervies_metrics(stage=sb.Stage.VALID, epoch=epoch, dataset=valid_set)

    def compute_forward(self, mix, targets, stage, noise=None):
        """Forward computations from the mixture to the separated signals."""

        # Unpack lists and put tensors in the right device
        mix, mix_lens = mix
        mix, mix_lens = mix.to(self.device), mix_lens.to(self.device)

        # Wrap target concatenation in a try/except block
        try:
            target_list = []
            for i in range(self.hparams.num_spks):
                t = targets[i][0].unsqueeze(-1)
                if t.numel() == 0:
                    print(f"[WARNING]  - Target tensor {i} is empty!")
                target_list.append(t)
            targets = torch.cat(target_list, dim=-1).to(self.device)
        except Exception as e:
            print(f"[ERROR] Failed to concatenate targets: {e}")

            # Create a directory to save error samples
            error_dir = "/export/fs05/afrumme1/sepformer_training/still_sonic_set_v1_reverb_sources_noisy_decomp_mix_results_metrics_log_errors"
            os.makedirs(error_dir, exist_ok=True)

            # Save mix waveform
            try:
                # Assuming mix is a 2D tensor [time, ...], we take the first channel if needed.
                mix_wave = mix[0].cpu()
                mix_filename = os.path.join(error_dir, f"error_mix.wav")
                torchaudio.save(mix_filename, mix_wave.unsqueeze(0), self.hparams.sample_rate)
                print(f"[INFO] Saved mix waveform for sample to {mix_filename}")
            except Exception as mix_e:
                print(f"[ERROR] Could not save mix waveform: {mix_e}")

            # Save each target waveform
            for i, t in enumerate(target_list):
                try:
                    target_wave = t.cpu()
                    # Remove the extra dimension (unsqueeze was added for concatenation)
                    target_filename = os.path.join(error_dir, f"target_{i}.wav")
                    # Ensure proper shape: [1, T]
                    torchaudio.save(target_filename, target_wave.squeeze(-1).unsqueeze(0), self.hparams.sample_rate)
                    print(f"[INFO] Saved target waveform for speaker {i} to {target_filename}")
                except Exception as target_e:
                    print(f"[ERROR] Could not save target waveform for speaker {i}: {target_e}")
            
            # Re-raise the exception so that the error is not silently ignored
            raise e

        # Add speech distortions
        if stage == sb.Stage.TRAIN:
            with torch.no_grad():
                if self.hparams.use_speedperturb:
                    mix, targets = self.add_speed_perturb(targets, mix_lens)

                    mix = targets.sum(-1)

                if self.hparams.use_wavedrop:
                    mix = self.hparams.drop_chunk(mix, mix_lens)
                    mix = self.hparams.drop_freq(mix)

                if self.hparams.limit_training_signal_len:
                    mix, targets = self.cut_signals(mix, targets)

        # Separation
        mix_w = self.hparams.Encoder(mix)
        est_mask = self.hparams.MaskNet(mix_w)
        mix_w = torch.stack([mix_w] * self.hparams.num_spks)
        sep_h = mix_w * est_mask

        # Decoding
        est_source = torch.cat(
            [
                self.hparams.Decoder(sep_h[i]).unsqueeze(-1)
                for i in range(self.hparams.num_spks)
            ],
            dim=-1,
        )

        # T changed after conv1d in encoder, fix it here
        T_origin = mix.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]

        return est_source, targets

    def compute_objectives(self, predictions, targets):
        """Computes the sinr loss"""
        return self.hparams.loss(targets, predictions)

    def fit_batch(self, batch):
        """Trains one batch"""
        amp = AMPConfig.from_name(self.precision)
        should_step = (self.step % self.grad_accumulation_factor) == 0

        # Unpacking batch list
        mixture = batch.mix_sig
        targets = [batch.s1_sig, batch.s2_sig]

        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)

        with self.no_sync(not should_step):
            if self.use_amp:
                with torch.autocast(
                    dtype=amp.dtype, device_type=torch.device(self.device).type
                ):
                    predictions, targets = self.compute_forward(
                        mixture, targets, sb.Stage.TRAIN
                    )
                    loss = self.compute_objectives(predictions, targets)

                    # hard threshold the easy dataitems
                    if self.hparams.threshold_byloss:
                        th = self.hparams.threshold
                        loss = loss[loss > th]
                        if loss.nelement() > 0:
                            loss = loss.mean()
                    else:
                        loss = loss.mean()

                if (
                    loss.nelement() > 0 and loss < self.hparams.loss_upper_lim
                ):  # the fix for computational problems
                    self.scaler.scale(loss).backward()
                    if self.hparams.clip_grad_norm >= 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.modules.parameters(),
                            self.hparams.clip_grad_norm,
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.nonfinite_count += 1
                    logger.info(
                        "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                            self.nonfinite_count
                        )
                    )
                    loss.data = torch.tensor(0.0).to(self.device)
            else:
                predictions, targets = self.compute_forward(
                    mixture, targets, sb.Stage.TRAIN
                )
                loss = self.compute_objectives(predictions, targets)

                if self.hparams.threshold_byloss:
                    th = self.hparams.threshold
                    loss = loss[loss > th]
                    if loss.nelement() > 0:
                        loss = loss.mean()
                else:
                    loss = loss.mean()

                if (
                    loss.nelement() > 0 and loss < self.hparams.loss_upper_lim
                ):  # the fix for computational problems
                    loss.backward()
                    if self.hparams.clip_grad_norm >= 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.modules.parameters(),
                            self.hparams.clip_grad_norm,
                        )
                    self.optimizer.step()
                else:
                    self.nonfinite_count += 1
                    logger.info(
                        "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                            self.nonfinite_count
                        )
                    )
                    loss.data = torch.tensor(0.0).to(self.device)
        self.optimizer.zero_grad()

        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        snt_id = batch.id
        mixture = batch.mix_sig
        targets = [batch.s1_sig, batch.s2_sig]
        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)

        with torch.no_grad():
            predictions, targets = self.compute_forward(mixture, targets, stage)
            loss = self.compute_objectives(predictions, targets)

        # Manage audio file saving
        if stage == sb.Stage.TEST and self.hparams.save_audio:
            if hasattr(self.hparams, "n_audio_to_save"):
                if self.hparams.n_audio_to_save > 0:
                    self.save_audio(snt_id[0], mixture, targets, predictions)
                    self.hparams.n_audio_to_save += -1
            else:
                self.save_audio(snt_id[0], mixture, targets, predictions)

        return loss.mean().detach()

    def compute_unseparation(self, predictions, stage):
        """
        Computes the unseparation metric based on the correlation between estimated sources.
        
        Args:
            predictions (torch.Tensor): The estimated separated sources (batch, time, num_spks).

        Returns:
            float: The correlation coefficient between separated tracks.
        """
        num_spks = predictions.shape[-1]

        if num_spks < 2:
            return 0.0  # Cannot compute correlation with less than two speakers

        unsep_values = []
        
        # Compute correlation between all speaker pairs
        for i in range(num_spks):
            for j in range(i + 1, num_spks):
                track1 = predictions[0, :, i].cpu().numpy()  # Convert to numpy
                track2 = predictions[0, :, j].cpu().numpy()
                
                # Compute correlation
                corr_matrix = np.corrcoef(track1, track2)
                correlation = corr_matrix[0, 1]
                unsep_values.append(correlation)  # Take absolute value for consistency
        
        # Average over all pairs
        return np.mean(unsep_values)

    def compute_metric(self, dataset, stage):
        """Computes multiple metrics for validation/test using estimated sources."""
        from mir_eval.separation import bss_eval_sources  # Import SDR computation

        all_sdrs = []
        all_unseparation_values = []
        
        # For ellipsis metric
        ellip_mse_list = []
        ellip_corr_list = []
        
        #window_sizes = [2048, 4096, 8192]
        window_sizes = [4096]

        # We'll keep a dictionary of lists, keyed by window_size.
        leakage_means_per_utterance = {ws: [] for ws in window_sizes}
        leakage_max_per_utterance   = {ws: [] for ws in window_sizes}
        # Load the validation/test dataset
        # If dataset is already a DataLoader or LoopedLoader, skip make_dataloader
        if isinstance(dataset, (sb.dataio.dataloader.SaveableDataLoader, sb.dataio.dataloader.LoopedLoader)):
            data_loader = dataset
        else:
            data_loader = sb.dataio.dataloader.make_dataloader(
                dataset, **self.hparams.dataloader_opts
            )

        with torch.no_grad():
            with tqdm(data_loader, dynamic_ncols=True) as t:
                for batch in t:
                    mixture = batch.mix_sig
                    targets = [batch.s1_sig, batch.s2_sig]
                    if self.hparams.num_spks == 3:
                        targets.append(batch.s3_sig)

                    # Get estimated sources
                    predictions, targets = self.compute_forward(mixture, targets, stage)

                    # Compute SI-SNR (already implemented)
                    sisnr = self.compute_objectives(predictions, targets)

                    # Compute SDR using mir_eval
                    sdr, _, _, _ = bss_eval_sources(
                        targets[0].t().cpu().numpy(),
                        predictions[0].t().detach().cpu().numpy(),
                    )

                    # Compute Unseparation Metric
                    unseparation_score = self.compute_unseparation(predictions, stage)

                    all_sdrs.append(sdr.mean())
                    all_unseparation_values.append(unseparation_score)
                    if predictions.shape[-1] >= 2:
                        # Take the first item in batch, speaker1, speaker2
                        pred_spk1 = predictions[0, :, 0].cpu().numpy()
                        pred_spk2 = predictions[0, :, 1].cpu().numpy()

                        for ws in window_sizes:
                            # get all window correlations
                            corrs = compute_leakage_for_pair(pred_spk1, pred_spk2, ws, stage)
                            if len(corrs) > 0:
                                leakage_means_per_utterance[ws].append(np.nanmean(corrs))
                                leakage_max_per_utterance[ws].append(np.nanmax(corrs))
                            else:
                                # If no valid window found, we can store 0 or np.nan
                                leakage_means_per_utterance[ws].append(np.nan)
                                leakage_max_per_utterance[ws].append(np.nan)
                                
                    mix_np = mixture[0][0].cpu().numpy()   # shape (T,)
                    # Grab the same separated signals used in leakage
                    if predictions.shape[-1] >= 2:
                        ellip_mse, ellip_corr, w = compute_ellipsis_metric(mix_np, pred_spk1, pred_spk2)
                        ellip_mse_list.append(ellip_mse)
                        ellip_corr_list.append(ellip_corr)
                    else:
                        # If <2 separated sources, store fallback
                        ellip_mse_list.append(np.nan)
                        ellip_corr_list.append(np.nan)
        # Compute averages
        avg_sdr = np.mean(all_sdrs)
        avg_unsep = np.mean(all_unseparation_values)
        avg_ellip_mse = np.nanmean(ellip_mse_list)
        avg_ellip_corr = np.nanmean(ellip_corr_list)
        leakage_stats = {}
        for ws in window_sizes:
            valid_means = [x for x in leakage_means_per_utterance[ws] if not np.isnan(x)]
            valid_maxes = [x for x in leakage_max_per_utterance[ws] if not np.isnan(x)]
            if len(valid_means) > 0:
                mean_leakage = np.mean(valid_means)
                max_leakage  = np.mean(valid_maxes)  # or maybe you want the overall max
            else:
                # If everything was silent, fallback to 0 or NaN
                mean_leakage = np.nan
                max_leakage = np.nan

            leakage_stats[ws] = {
                "mean_leakage": mean_leakage,
                "max_leakage":  max_leakage,
            }
            
        # Console logs
        if sb.utils.distributed.if_main_process():
            logger.info(f"{stage} - Mean SDR: {avg_sdr:.4f}, Mean Unseparation: {avg_unsep:.4f}")
            logger.info(f"{stage} - Ellipsis MSE: {avg_ellip_mse:.6f}, Corr: {avg_ellip_corr:.6f}")

        for ws in window_sizes:
            mean_val = leakage_stats[ws]["mean_leakage"]
            max_val  = leakage_stats[ws]["max_leakage"]
            if sb.utils.distributed.if_main_process():
                logger.info(
                    f"{stage} - Window size {ws}: "
                    f"Mean leakage = {mean_val if not np.isnan(mean_val) else 'NaN'}, "
                    f"Max leakage = {max_val if not np.isnan(max_val) else 'NaN'}"
                )
        metrics_dict = {
        "sdr":         float(avg_sdr),
        "unseparation": float(avg_unsep),
        "ellip_mse":   float(avg_ellip_mse),
        "ellip_corr":  float(avg_ellip_corr),
        "leakage":     leakage_stats
        }
        return metrics_dict  # You can return both if needed
    
    def log_unsupervies_metrics(self, stage, epoch, dataset=None):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {}
            # Compute custom metrics for validation and test stages
        #if (stage == sb.Stage.VALID or stage == sb.Stage.TEST) and dataset is not None:
        if True:
            all_metrics = self.compute_metric(dataset, stage)
            # Merge relevant keys
            stage_stats["sdr"] = all_metrics["sdr"]
            stage_stats["unseparation"] = all_metrics["unseparation"]
            stage_stats["ellip_mse"] = all_metrics["ellip_mse"]
            stage_stats["ellip_corr"] = all_metrics["ellip_corr"]

            # If you want to flatten out the leakage stats:
            for ws, vals in all_metrics["leakage"].items():
                stage_stats[f"leakage_{ws}_mean"] = vals["mean_leakage"]
                stage_stats[f"leakage_{ws}_max"]  = vals["max_leakage"]
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Perform end-of-iteration things, like annealing, logging, etc.
        # Only the main process should log and save checkpoints
        if sb.utils.distributed.if_main_process():  
            if stage == sb.Stage.VALID:
                # Learning rate annealing
                self.hparams.train_logger.log_stats(
                    stats_meta={"epoch": epoch},
                    valid_stats=stage_stats
                )
                # self.checkpointer.save_and_keep_only(
                #     meta={"si-snr": stage_stats["si-snr"]}, min_keys=["si-snr"]
                # )
            elif stage == sb.Stage.TEST:
                self.hparams.train_logger.log_stats(
                    stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                    test_stats=stage_stats
                )

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"si-snr": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch},
                train_stats=self.train_stats
            )

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            # Learning rate annealing
            if isinstance(
                self.hparams.lr_scheduler, schedulers.ReduceLROnPlateau
            ):
                current_lr, next_lr = self.hparams.lr_scheduler(
                    [self.optimizer], epoch, stage_loss
                )
                schedulers.update_learning_rate(self.optimizer, next_lr)
            else:
                # if we do not use the reducelronplateau, we do not change the lr
                current_lr = self.hparams.optimizer.optim.param_groups[0]["lr"]
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": current_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"si-snr": stage_stats["si-snr"]}, min_keys=["si-snr"]
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            
    def add_speed_perturb(self, targets, targ_lens):
        """Adds speed perturbation and random_shift to the input signals"""

        min_len = -1
        recombine = False

        if self.hparams.use_speedperturb or self.hparams.use_rand_shift:
            # Performing speed change (independently on each source)
            new_targets = []
            recombine = True

            for i in range(targets.shape[-1]):
                new_target = self.hparams.speed_perturb(targets[:, :, i])
                new_targets.append(new_target)
                if i == 0:
                    min_len = new_target.shape[-1]
                else:
                    if new_target.shape[-1] < min_len:
                        min_len = new_target.shape[-1]

            if self.hparams.use_rand_shift:
                # Performing random_shift (independently on each source)
                recombine = True
                for i in range(targets.shape[-1]):
                    rand_shift = torch.randint(
                        self.hparams.min_shift, self.hparams.max_shift, (1,)
                    )
                    new_targets[i] = new_targets[i].to(self.device)
                    new_targets[i] = torch.roll(
                        new_targets[i], shifts=(rand_shift[0],), dims=1
                    )

            # Re-combination
            if recombine:
                if self.hparams.use_speedperturb:
                    targets = torch.zeros(
                        targets.shape[0],
                        min_len,
                        targets.shape[-1],
                        device=targets.device,
                        dtype=torch.float,
                    )
                for i, new_target in enumerate(new_targets):
                    targets[:, :, i] = new_targets[i][:, 0:min_len]

        mix = targets.sum(-1)
        return mix, targets

    def cut_signals(self, mixture, targets):
        """This function selects a random segment of a given length within the mixture.
        The corresponding targets are selected accordingly"""
        randstart = torch.randint(
            0,
            1 + max(0, mixture.shape[1] - self.hparams.training_signal_len),
            (1,),
        ).item()
        targets = targets[
            :, randstart : randstart + self.hparams.training_signal_len, :
        ]
        mixture = mixture[
            :, randstart : randstart + self.hparams.training_signal_len
        ]
        return mixture, targets

    def reset_layer_recursively(self, layer):
        """Reinitializes the parameters of the neural networks"""
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        for child_layer in layer.modules():
            if layer != child_layer:
                self.reset_layer_recursively(child_layer)

    def save_results(self, test_data):
        """This script computes the SDR and SI-SNR metrics and saves
        them into a csv file"""

        # This package is required for SDR computation
        from mir_eval.separation import bss_eval_sources

        # Create folders where to store audio
        save_file = os.path.join(self.hparams.output_folder, "test_results.csv")

        # Variable init
        all_sdrs = []
        all_sdrs_i = []
        all_sisnrs = []
        all_sisnrs_i = []
        csv_columns = ["snt_id", "sdr", "sdr_i", "si-snr", "si-snr_i"]

        test_loader = sb.dataio.dataloader.make_dataloader(
            test_data, **self.hparams.dataloader_opts
        )

        with open(save_file, "w", newline="", encoding="utf-8") as results_csv:
            writer = csv.DictWriter(results_csv, fieldnames=csv_columns)
            writer.writeheader()

            # Loop over all test sentence
            with tqdm(test_loader, dynamic_ncols=True) as t:
                for i, batch in enumerate(t):
                    # Apply Separation
                    mixture, mix_len = batch.mix_sig
                    snt_id = batch.id
                    targets = [batch.s1_sig, batch.s2_sig]
                    if self.hparams.num_spks == 3:
                        targets.append(batch.s3_sig)

                    with torch.no_grad():
                        predictions, targets = self.compute_forward(
                            batch.mix_sig, targets, sb.Stage.TEST
                        )

                    # Compute SI-SNR
                    sisnr = self.compute_objectives(predictions, targets)

                    # Compute SI-SNR improvement
                    mixture_signal = torch.stack(
                        [mixture] * self.hparams.num_spks, dim=-1
                    )
                    mixture_signal = mixture_signal.to(targets.device)
                    sisnr_baseline = self.compute_objectives(
                        mixture_signal, targets
                    )
                    sisnr_i = sisnr - sisnr_baseline

                    # Compute SDR
                    sdr, _, _, _ = bss_eval_sources(
                        targets[0].t().cpu().numpy(),
                        predictions[0].t().detach().cpu().numpy(),
                    )

                    sdr_baseline, _, _, _ = bss_eval_sources(
                        targets[0].t().cpu().numpy(),
                        mixture_signal[0].t().detach().cpu().numpy(),
                    )

                    sdr_i = sdr.mean() - sdr_baseline.mean()

                    # Saving on a csv file
                    row = {
                        "snt_id": snt_id[0],
                        "sdr": sdr.mean(),
                        "sdr_i": sdr_i,
                        "si-snr": -sisnr.item(),
                        "si-snr_i": -sisnr_i.item(),
                    }
                    writer.writerow(row)

                    # Metric Accumulation
                    all_sdrs.append(sdr.mean())
                    all_sdrs_i.append(sdr_i.mean())
                    all_sisnrs.append(-sisnr.item())
                    all_sisnrs_i.append(-sisnr_i.item())

                row = {
                    "snt_id": "avg",
                    "sdr": np.array(all_sdrs).mean(),
                    "sdr_i": np.array(all_sdrs_i).mean(),
                    "si-snr": np.array(all_sisnrs).mean(),
                    "si-snr_i": np.array(all_sisnrs_i).mean(),
                }
                writer.writerow(row)
        if sb.utils.distributed.if_main_process():
            logger.info("Mean SISNR is {}".format(np.array(all_sisnrs).mean()))
            logger.info("Mean SISNRi is {}".format(np.array(all_sisnrs_i).mean()))
            logger.info("Mean SDR is {}".format(np.array(all_sdrs).mean()))
            logger.info("Mean SDRi is {}".format(np.array(all_sdrs_i).mean()))

    def save_audio(self, snt_id, mixture, targets, predictions):
        "saves the test audio (mixture, targets, and estimated sources) on disk"

        # Create output folder
        save_path = os.path.join(self.hparams.save_folder, "audio_results")
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for ns in range(self.hparams.num_spks):
            # Estimated source
            signal = predictions[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, "item{}_source{}hat.wav".format(snt_id, ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )

            # Original source
            signal = targets[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, "item{}_source{}.wav".format(snt_id, ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )

        # Mixture
        signal = mixture[0][0, :]
        signal = signal / signal.abs().max()
        save_file = os.path.join(save_path, "item{}_mix.wav".format(snt_id))
        torchaudio.save(
            save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
        )


def dataio_prep(hparams):
    """Creates data processing pipeline"""

    # 1. Define datasets
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    datasets = [train_data, valid_data, test_data]

    # 2. Provide audio pipelines

    @sb.utils.data_pipeline.takes("mix_wav")
    @sb.utils.data_pipeline.provides("mix_sig")
    def audio_pipeline_mix(mix_wav):
        mix_sig = sb.dataio.dataio.read_audio(mix_wav)
        return mix_sig

    @sb.utils.data_pipeline.takes("s1_wav")
    @sb.utils.data_pipeline.provides("s1_sig")
    def audio_pipeline_s1(s1_wav):
        s1_sig = sb.dataio.dataio.read_audio(s1_wav)
        return s1_sig

    @sb.utils.data_pipeline.takes("s2_wav")
    @sb.utils.data_pipeline.provides("s2_sig")
    def audio_pipeline_s2(s2_wav):
        s2_sig = sb.dataio.dataio.read_audio(s2_wav)
        return s2_sig

    if hparams["num_spks"] == 3:

        @sb.utils.data_pipeline.takes("s3_wav")
        @sb.utils.data_pipeline.provides("s3_sig")
        def audio_pipeline_s3(s3_wav):
            s3_sig = sb.dataio.dataio.read_audio(s3_wav)
            return s3_sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_mix)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s1)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s2)
    if hparams["num_spks"] == 3:
        sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s3)
        sb.dataio.dataset.set_output_keys(
            datasets, ["id", "mix_sig", "s1_sig", "s2_sig", "s3_sig"]
        )
    else:
        sb.dataio.dataset.set_output_keys(
            datasets, ["id", "mix_sig", "s1_sig", "s2_sig"]
        )

    return train_data, valid_data, test_data

def check_sample(idx, dataset, num_spks):
    """
    Checks a single sample for problematic targets.
    
    Returns:
      - None if the sample is OK.
      - A tuple (idx, spk_idx) if the target for speaker spk_idx is empty.
      - A tuple (idx, "missing key", key) if an expected key is missing.
      - A tuple (idx, "exception", str(e)) if any other exception occurs.
    """
    try:
        sample = dataset[idx]
        # Iterate over expected number of speakers (assuming keys are s1_sig, s2_sig, etc.)
        for i in range(num_spks):
            key = f's{i+1}_sig'
            if key not in sample:
                return (idx, "missing key", key)
            target = sample[key]
            if target.numel() == 0:
                return (idx, i)
        return None  # sample is valid
    except Exception as e:
        return (idx, "exception", str(e))

def scan_problematic_samples(dataset, num_spks, num_workers=4):
    """
    Scans the entire dataset using a ThreadPoolExecutor.
    
    Returns a list of problematic sample indicators.
    """
    problematic_samples = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(check_sample, idx, dataset, num_spks): idx
            for idx in range(len(dataset))
        }
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Scanning samples"):
            result = future.result()
            if result is not None:
                print(f"[WARNING] Problematic sample found: {result}",flush=True)
                problematic_samples.append(result)
    return problematic_samples


if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Logger info
    logger = get_logger(__name__)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Update precision to bf16 if the device is CPU and precision is fp16
    if run_opts.get("device") == "cpu" and hparams.get("precision") == "fp16":
        hparams["precision"] = "bf16"

    # Check if wsj0_tr is set with dynamic mixing
    if hparams["dynamic_mixing"] and not os.path.exists(
        hparams["base_folder_dm"]
    ):
        raise ValueError(
            "Please, specify a valid base_folder_dm folder when using dynamic mixing"
        )

    # Data preparation
    from prepare_data import prepare_wsjmix  # noqa

    run_on_main(
        prepare_wsjmix,
        kwargs={
            "datapath": hparams["data_folder"],
            "savepath": hparams["save_folder"],
            "n_spks": hparams["num_spks"],
            "skip_prep": hparams["skip_prep"],
            "fs": hparams["sample_rate"],
        },
    )

    # Create dataset objects
    if hparams["dynamic_mixing"]:
        from dynamic_mixing import dynamic_mix_data_prep

        # if the base_folder for dm is not processed, preprocess them
        if "processed" not in hparams["base_folder_dm"]:
            # if the processed folder already exists we just use it otherwise we do the preprocessing
            if not os.path.exists(
                os.path.normpath(hparams["base_folder_dm"]) + "_processed"
            ):
                from preprocess_dynamic_mixing import resample_folder

                print("Resampling the base folder")
                run_on_main(
                    resample_folder,
                    kwargs={
                        "input_folder": hparams["base_folder_dm"],
                        "output_folder": os.path.normpath(
                            hparams["base_folder_dm"]
                        )
                        + "_processed",
                        "fs": hparams["sample_rate"],
                        "regex": "**/*.wav",
                    },
                )
                # adjust the base_folder_dm path
                hparams["base_folder_dm"] = (
                    os.path.normpath(hparams["base_folder_dm"]) + "_processed"
                )
            else:
                print(
                    "Using the existing processed folder on the same directory as base_folder_dm"
                )
                hparams["base_folder_dm"] = (
                    os.path.normpath(hparams["base_folder_dm"]) + "_processed"
                )

        # Collecting the hparams for dynamic batching
        dm_hparams = {
            "train_data": hparams["train_data"],
            "data_folder": hparams["data_folder"],
            "base_folder_dm": hparams["base_folder_dm"],
            "sample_rate": hparams["sample_rate"],
            "num_spks": hparams["num_spks"],
            "training_signal_len": hparams["training_signal_len"],
            "dataloader_opts": hparams["dataloader_opts"],
        }
        train_data = dynamic_mix_data_prep(dm_hparams)
        _, valid_data, test_data = dataio_prep(hparams)
    else:
        train_data, valid_data, test_data = dataio_prep(hparams)
        print("in else - no dynamic mixing")


    # # Example: Print a sample from train_data for debugging.
    # print(f"train_data[0]: {train_data[0]}")
    # print(f"Original training set size: {len(train_data)}")
    
    # print("Scanning training data for problematic samples using multiprocessing...")
    # bad_samples_train = scan_problematic_samples(train_data, hparams["num_spks"], num_workers=8)
    # if bad_samples_train:
    #     print(f"Found {len(bad_samples_train)} problematic samples in the training set:")
    #     for item in bad_samples_train:
    #         print(item)
    # else:
    #     print("No problematic samples found in the training set.")
    
    # print("Scanning validation data for problematic samples...")
    # bad_samples_valid = scan_problematic_samples(valid_data, hparams["num_spks"], num_workers=8)
    # if bad_samples_valid:
    #     print(f"Found {len(bad_samples_valid)} problematic samples in the validation set:")
    #     for item in bad_samples_valid:
    #         print(item)
    # else:
    #     print("No problematic samples found in the validation set.")
    
    # print("Scanning test data for problematic samples...")
    # bad_samples_test = scan_problematic_samples(test_data, hparams["num_spks"], num_workers=8)
    # if bad_samples_test:
    #     print(f"Found {len(bad_samples_test)} problematic samples in the test set:")
    #     for item in bad_samples_test:
    #         print(item)
    # else:
    #     print("No problematic samples found in the test set.")
        
    # Load pretrained model if pretrained_separator is present in the yaml
    if "pretrained_separator" in hparams:
        run_on_main(hparams["pretrained_separator"].collect_files)
        hparams["pretrained_separator"].load_collected()

    # Brain class initialization
    separator = Separation(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # re-initialize the parameters if we don't use a pretrained model
    if "pretrained_separator" not in hparams:
        for module in separator.modules.values():
            separator.reset_layer_recursively(module)

    # Training
    separator.fit(
        separator.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_opts"],
        valid_loader_kwargs=hparams["dataloader_opts"],
    )

    # Eval
    separator.evaluate(test_data, min_key="si-snr")
    separator.save_results(test_data)
