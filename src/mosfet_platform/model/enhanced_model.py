from dataclasses import dataclass

import numpy as np

from mosfet_platform.model.current_components import (
    calculate_subthreshold_current,
    calculate_strong_inversion_current,
    combine_current_components,
)
from mosfet_platform.model.mobility import mobility_model
from mosfet_platform.model.temperature import thermal_voltage

EPSILON_0 = 8.854187817e-12
EPSILON_OX = 3.9


@dataclass
class EnhancedModelParameters:
    w: float
    l: float
    tox: float
    mu: float
    vth: float
    lambda_clm: float
    subthreshold_n: float = 1.5
    temperature: float = 300.0
    i0: float = 1e-6
    dibl_coeff: float = 0.05
    l_ref: float = 180e-9
    theta_mobility: float = 0.2

    def __post_init__(self) -> None:
        if self.w <= 0 or self.l <= 0:
            raise ValueError("Channel dimensions must be positive.")
        if self.tox <= 0:
            raise ValueError("Oxide thickness must be positive.")
        if self.mu <= 0:
            raise ValueError("Mobility must be positive.")
        if self.lambda_clm < 0:
            raise ValueError("Channel-length modulation must be non-negative.")
        if self.subthreshold_n < 1:
            raise ValueError("Subthreshold factor must be at least 1.")
        if self.temperature <= 0:
            raise ValueError("Temperature must be positive.")
        if self.dibl_coeff < 0:
            raise ValueError("DIBL coefficient must be non-negative.")
        if self.l_ref <= 0:
            raise ValueError("Reference channel length l_ref must be positive.")
        if self.theta_mobility < 0:
            raise ValueError("Mobility degradation coefficient must be non-negative.")


def calculate_cox(tox: float) -> float:
    if tox <= 0:
        raise ValueError("Oxide thickness must be positive.")
    return EPSILON_OX * EPSILON_0 / tox

def effective_vth_dibl(
    vds: float | np.ndarray,
    params: EnhancedModelParameters,
) -> float | np.ndarray:
    """
    Vth_eff = Vth0 - dibl_coeff * Vds * (Lref / L)
    """
    vds_arr = np.asarray(vds, dtype=float)

    if np.any(vds_arr < 0):
        raise ValueError("Vds must be non-negative for this simplified model.")

    vth_eff = params.vth - params.dibl_coeff * vds_arr * (params.l_ref / params.l)

    if np.isscalar(vds):
        return float(vth_eff)

    return vth_eff

def ids_nmos_enhanced(vgs, vds, params):
    vgs_arr = np.asarray(vgs, dtype=float)
    vds_arr = np.asarray(vds, dtype=float)

    cox = calculate_cox(params.tox)
    vt = thermal_voltage(params.temperature)

    vth_eff = effective_vth_dibl(vds=vds_arr, params=params)
    vov = vgs_arr - vth_eff
    vov_pos = np.maximum(vov, 0.0)

    mu_eff = mobility_model(
        mu0=params.mu,
        vov=vov_pos,
        theta=params.theta_mobility,
    )

    beta_eff = mu_eff * cox * (params.w / params.l)

    id_sub = calculate_subthreshold_current(
        vov=vov,
        vds=vds_arr,
        vt=vt,
        i0=params.i0,
        w=params.w,
        l=params.l,
        subthreshold_n=params.subthreshold_n,
    )

    id_strong = calculate_strong_inversion_current(
        vov=vov,
        vds=vds_arr,
        beta=beta_eff,
        lambda_clm=params.lambda_clm,
    )

    ids = combine_current_components(
        id_sub=id_sub,
        id_strong=id_strong,
    )

    if np.isscalar(vgs) and np.isscalar(vds):
        return float(ids)

    return ids
