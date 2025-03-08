import torch
import torch.nn as nn
from speechbrain.nnet.losses import get_si_snr_with_pitwrapper

class WeightedSiSNRCorrelation(nn.Module):
    """
    Custom loss combining SI-SNR (with PIT) and a negative correlation penalty
    between the separated sources.

    The loss is computed as:
        loss = alpha * SI-SNR_loss + (1 - alpha) * (-correlation)
    
    where SI-SNR_loss is computed using SpeechBrain's get_si_snr_with_pitwrapper, and
    the correlation is the Pearson correlation coefficient between the two estimated sources.
    
    This implementation assumes that the input tensors have shape [B, T, C],
    with C=2 for two speakers.

    Args:
        alpha (float): Weight for the SI-SNR loss term. The correlation term is weighted by (1 - alpha).
    """
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, targets, estimated):
        """
        Forward pass to compute the weighted SI-SNR and correlation loss.
        
        Args:
            targets (Tensor): Ground truth source signals, shape [B, T, C].
            estimated (Tensor): Estimated source signals, shape [B, T, C].
        
        Returns:
            loss (Tensor): The computed loss as a scalar.
        """
        # Compute SI-SNR loss using SpeechBrain's PIT wrapper.
        si_snr_loss = get_si_snr_with_pitwrapper(targets, estimated)

        # Ensure the input is for two-speaker separation (i.e., last dimension is 2)
        if estimated.size(-1) != 2:
            raise ValueError("WeightedSiSNRCorrelation is designed for two-speaker separation (last dimension should be 2).")
        
        # Extract the two estimated sources
        x1 = estimated[..., 0]  # shape: [B, T]
        x2 = estimated[..., 1]  # shape: [B, T]

        # Compute the Pearson correlation coefficient for each sample.
        x1_mean = x1.mean(dim=1, keepdim=True)
        x2_mean = x2.mean(dim=1, keepdim=True)
        x1_centered = x1 - x1_mean
        x2_centered = x2 - x2_mean

        numerator = (x1_centered * x2_centered).sum(dim=1)
        # Add a small epsilon only once to ensure numerical stability.
        denominator = torch.sqrt((x1_centered ** 2).sum(dim=1) * (x2_centered ** 2).sum(dim=1)) + 1e-8
        correlation = numerator / denominator

        # We want to penalize high correlation (i.e. encourage uncorrelated sources),
        # so we take the negative of the average correlation.
        corr_loss = correlation.mean()

        # Combine the SI-SNR loss and the negative correlation loss
        loss = self.alpha * si_snr_loss + (1 - self.alpha) * corr_loss

        return loss


class WeightedSiSNRLeakage(nn.Module):
    """
    Custom loss combining SI-SNR (with PIT) and a leakage metric computed over
    sliding windows of half a second (4000 samples at 8000 Hz).

    The loss is computed as:
        loss = alpha * SI-SNR_loss + (1 - alpha) * leakage_metric

    where:
      - SI-SNR_loss is computed using SpeechBrain's get_si_snr_with_pitwrapper.
      - leakage_metric is the mean Pearson correlation computed over sliding windows
        between the two estimated sources. A higher leakage (i.e. higher correlation)
        indicates more signal leakage between channels.

    This implementation assumes that the input tensors have shape [B, T, C] with C=2
    for two speakers.

    Args:
        alpha (float): Weight for the SI-SNR loss term. The leakage term is weighted by (1 - alpha).
        window_size (int): Number of samples per window (default 4000 for 0.5 seconds at 8000 Hz).
        silence_rms_threshold (float): RMS threshold to mark a window as silent.
        mean_amplitude_threshold (float): Mean amplitude threshold to mark a window as silent.
    """
    def __init__(self, alpha=0.5, window_size=4000, silence_rms_threshold=1e-4, mean_amplitude_threshold=1e-4):
        super().__init__()
        self.alpha = alpha
        self.window_size = window_size
        self.window_step = window_size // 2  # 50% overlap
        self.silence_rms_threshold = silence_rms_threshold
        self.mean_amplitude_threshold = mean_amplitude_threshold

    def compute_leakage_vectorized(self, x1, x2):
        """
        Vectorized computation of the leakage metric for a pair of signals.

        Args:
            x1 (Tensor): Estimated source 1, shape [B, T].
            x2 (Tensor): Estimated source 2, shape [B, T].

        Returns:
            leakage (Tensor): A scalar tensor containing the mean leakage (correlation)
                              over non-silent windows across the batch.
        """
        # Unfold the signals into sliding windows: shape -> [B, num_windows, window_size]
        windows1 = x1.unfold(dimension=1, size=self.window_size, step=self.window_step)
        windows2 = x2.unfold(dimension=1, size=self.window_size, step=self.window_step)

        # Compute per-window statistics
        # RMS and average amplitude for silence detection:
        win1_rms = torch.sqrt((windows1 ** 2).mean(dim=-1))
        win2_rms = torch.sqrt((windows2 ** 2).mean(dim=-1))
        win1_avg = windows1.abs().mean(dim=-1)
        win2_avg = windows2.abs().mean(dim=-1)

        # Create a valid-window mask: a window is valid if neither channel is silent
        valid_mask = ~(((win1_rms < self.silence_rms_threshold) & (win1_avg < self.mean_amplitude_threshold)) |
                       ((win2_rms < self.silence_rms_threshold) & (win2_avg < self.mean_amplitude_threshold)))
        # Also require non-zero standard deviation to avoid division by zero
        win1_std = windows1.std(dim=-1)
        win2_std = windows2.std(dim=-1)
        valid_mask = valid_mask & (win1_std > 0) & (win2_std > 0)

        # Compute centered windows
        win1_mean = windows1.mean(dim=-1, keepdim=True)
        win2_mean = windows2.mean(dim=-1, keepdim=True)
        win1_centered = windows1 - win1_mean
        win2_centered = windows2 - win2_mean

        # Compute Pearson correlation for each window:
        numerator = (win1_centered * win2_centered).sum(dim=-1)
        denominator = torch.sqrt((win1_centered ** 2).sum(dim=-1) * (win2_centered ** 2).sum(dim=-1)) + 1e-8
        corr = numerator / denominator  # shape [B, num_windows]

        # Apply valid mask: set invalid windows to zero so they don't contribute.
        corr = corr * valid_mask.float()

        # Compute the mean correlation per sample, taking into account only valid windows.
        valid_counts = valid_mask.float().sum(dim=-1)  # shape [B]
        # Avoid division by zero by replacing zero counts with 1 (the corresponding sum is zero)
        mean_corr_per_sample = torch.where(valid_counts > 0,
                                           corr.sum(dim=-1) / valid_counts,
                                           torch.zeros_like(valid_counts))
        # Return the average leakage over the batch.
        return mean_corr_per_sample.mean()

    def forward(self, targets, estimated):
        """
        Forward pass to compute the combined loss.

        Args:
            targets (Tensor): Ground truth source signals, shape [B, T, C].
            estimated (Tensor): Estimated source signals, shape [B, T, C].

        Returns:
            loss (Tensor): The combined loss as a scalar.
        """
        # Compute SI-SNR loss using SpeechBrain's PIT wrapper.
        si_snr_loss = get_si_snr_with_pitwrapper(targets, estimated)

        # Ensure the input is for two-speaker separation (i.e., last dimension is 2)
        if estimated.size(-1) != 2:
            raise ValueError("WeightedSiSNRLeakage is designed for two-speaker separation (last dimension should be 2).")
        
        # Extract the two estimated sources: shape [B, T]
        x1 = estimated[..., 0]
        x2 = estimated[..., 1]

        # Compute the leakage metric vectorized over the batch.
        leakage_metric = self.compute_leakage_vectorized(x1, x2)

        # Combine the SI-SNR loss and the leakage metric with weighting.
        loss = self.alpha * si_snr_loss + (1 - self.alpha) * leakage_metric
        return loss