import numpy as np

def smooth_positive(x, alpha: float = 0.02):
    x_arr = np.asarray(x, dtype=float)
    return alpha * np.log1p(np.exp(x_arr / alpha))

def transition_weight(x, alpha: float = 0.03):
    x_arr = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x_arr / alpha))