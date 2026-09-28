import numpy as np

def mobility_model(
    mu0 : float,
    vov: float | np.ndarray,
    theta: float,
)-> float | np.ndarray:
    if mu0 <= 0:
        raise ValueError("Low-field mobility mu0 must be positive.")
    if theta < 0:
        raise ValueError("Mobility degradation coefficient theta must be non-negative.")
    vov_arr = np.asarray(vov, dtype=float)
    vov_pos = np.maximum(vov_arr, 0.0)

    mu_eff = mu0 / (1.0 + theta * vov_pos)

    if np.isscalar(vov):
        return float(mu_eff)

    return mu_eff