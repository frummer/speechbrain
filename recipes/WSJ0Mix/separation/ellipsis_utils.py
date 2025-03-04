import numpy as np

def compute_ellipsis_metric(mixture: np.ndarray, sep1: np.ndarray, sep2: np.ndarray):
    """
    Computes how well a linear combination of sep1 and sep2
    can reconstruct `mixture`. 
    Returns:
       mse (float): Mean-squared error between mixture and reconstructed mixture.
       corr (float): Correlation between mixture and its reconstruction.
       w (ndarray): The 2x1 vector of linear weights found by lstsq.
    """
    # Force same length
    min_len = min(len(sep1), len(sep2), len(mixture))
    mixture = mixture[:min_len]
    sep1 = sep1[:min_len]
    sep2 = sep2[:min_len]

    # Make matrix S with shape (T, 2)
    S = np.column_stack((sep1, sep2))

    # Solve least squares
    w, residuals, rank, s_vals = np.linalg.lstsq(S, mixture, rcond=None)
    if rank < 2:
        # If rank<2, it means the two separated tracks are effectively linearly dependent.
        # Return some fallback. E.g. MSE=1.0, correlation=0.0
        return 1.0, 0.0, w

    # Reconstructed mixture
    m_hat = S @ w

    # MSE
    mse = np.mean((mixture - m_hat) ** 2)

    # correlation
    corr = np.corrcoef(mixture, m_hat)[0, 1]

    return mse, corr, w
