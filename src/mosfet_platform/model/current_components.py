import numpy as np
from mosfet_platform.model.smoothing import smooth_positive

def calculate_subthreshold_current(
    vov: float | np.ndarray,
    vds: float | np.ndarray,
    vt: float,
    i0: float,
    w: float,
    l: float,
    subthreshold_n: float,
) -> float | np.ndarray:
    vov_arr = np.asarray(vov, dtype=float)
    vds_arr = np.asarray(vds, dtype=float)

    vov_sub = np.minimum(vov_arr, 0.0)
    vds_factor = 1.0 - np.exp(-vds_arr / vt)

    return (i0* (w / l)* np.exp(vov_sub / (subthreshold_n * vt))* vds_factor)

def calculate_strong_inversion_current(
    vov: float | np.ndarray,
    vds: float | np.ndarray,
    beta: float | np.ndarray,
    lambda_clm: float,
    smoothing_alpha: float = 0.02,
) -> float | np.ndarray:
    vov_arr = np.asarray(vov, dtype=float)
    vds_arr = np.asarray(vds, dtype=float)

    vov_pos = smooth_positive(vov_arr, alpha=smoothing_alpha)

    id_linear = beta * (vov_pos * vds_arr - 0.5 * vds_arr**2)
    id_linear = id_linear * (1.0 + lambda_clm * vds_arr)

    id_sat = 0.5 * beta * vov_pos**2 * (1.0 + lambda_clm * vds_arr)

    linear_region = (vov_pos > 0.0) & (vds_arr < vov_pos)

    return np.where(linear_region, id_linear, id_sat)

def combine_current_components(
    id_sub: float | np.ndarray,
    id_strong: float | np.ndarray,
) -> float | np.ndarray:
    ids = np.asarray(id_sub) + np.asarray(id_strong)
    return np.maximum(ids, 0.0)