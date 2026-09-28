from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from mosfet_platform.model.enhanced_model import (
    EnhancedModelParameters,
    ids_nmos_enhanced,
)

@dataclass(frozen=True)
class GeometryEnvelope:
    """Declared geometry range for model prediction."""
    length_min_m: float
    length_max_m: float
    tox_min_m: float
    tox_max_m: float

    def __post_init__(self) -> None:
        if not np.isfinite((self.length_min_m, self.length_max_m, self.tox_min_m, self.tox_max_m)).all():
            raise ValueError("Geometry envelope bounds must be finite.")
        if self.length_min_m <= 0.0 or self.tox_min_m <= 0.0:
            raise ValueError("Geometry envelope minima must be positive.")
        if self.length_max_m < self.length_min_m:
            raise ValueError("Geometry envelope length bounds are reversed.")
        if self.tox_max_m < self.tox_min_m:
            raise ValueError("Geometry envelope tox bounds are reversed.")

    def contains(self, length_m: float, tox_m: float) -> bool:
        length_tolerance = max(abs(self.length_max_m), 1.0) * 1.0e-15
        tox_tolerance = max(abs(self.tox_max_m), 1.0) * 1.0e-15
        return bool(
            self.length_min_m - length_tolerance
            <= float(length_m)
            <= self.length_max_m + length_tolerance
            and self.tox_min_m - tox_tolerance
            <= float(tox_m)
            <= self.tox_max_m + tox_tolerance
        )

    def require_contains(self, length_m: float, tox_m: float) -> None:
        if not self.contains(length_m, tox_m):
            raise ValueError(
                "Requested geometry is outside the qualified geometry envelope: "
                f"Lg={float(length_m):.6g} m, tox={float(tox_m):.6g} m."
            )


@dataclass(frozen=True)
class GeometryAwareModelParameters:
    width_m: float
    temperature_K: float
    length_ref_m: float
    tox_ref_m: float
    theta_mobility: float
    vth_ref_V: float
    mu_ref_m2_per_Vs: float
    subthreshold_n_ref: float
    i0_A: float
    dibl_ref_V_per_V: float
    lambda_ref_1_per_V: float
    vth_length_slope_V: float
    vth_tox_slope_V: float
    log_mu_length_slope: float
    log_mu_tox_slope: float
    log_n_minus_one_length_slope: float
    log_n_minus_one_tox_slope: float
    log_dibl_length_slope: float
    log_dibl_tox_slope: float
    log_lambda_length_slope: float
    log_lambda_tox_slope: float
    envelope: GeometryEnvelope
    geometry_family: str = "additive"
    vth_interaction_slope_V: float = 0.0
    log_mu_interaction_slope: float = 0.0
    log_n_minus_one_interaction_slope: float = 0.0
    log_dibl_interaction_slope: float = 0.0
    log_lambda_interaction_slope: float = 0.0

    def __post_init__(self) -> None:
        positive = {
            "width_m": self.width_m,
            "temperature_K": self.temperature_K,
            "length_ref_m": self.length_ref_m,
            "tox_ref_m": self.tox_ref_m,
            "mu_ref_m2_per_Vs": self.mu_ref_m2_per_Vs,
            "i0_A": self.i0_A,
            "dibl_ref_V_per_V": self.dibl_ref_V_per_V,
            "lambda_ref_1_per_V": self.lambda_ref_1_per_V,
        }
        for name, value in positive.items():
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if not np.isfinite(float(self.vth_ref_V)):
            raise ValueError("vth_ref_V must be finite.")
        if self.subthreshold_n_ref < 1.0 or not np.isfinite(
            float(self.subthreshold_n_ref)
        ):
            raise ValueError("subthreshold_n_ref must be finite and at least 1.")
        if self.theta_mobility < 0.0 or not np.isfinite(
            float(self.theta_mobility)
        ):
            raise ValueError("theta_mobility must be finite and non-negative.")
        if self.geometry_family not in {"additive", "interaction"}:
            raise ValueError("geometry_family must be additive or interaction.")
        slope_names = (
            "vth_length_slope_V",
            "vth_tox_slope_V",
            "log_mu_length_slope",
            "log_mu_tox_slope",
            "log_n_minus_one_length_slope",
            "log_n_minus_one_tox_slope",
            "log_dibl_length_slope",
            "log_dibl_tox_slope",
            "log_lambda_length_slope",
            "log_lambda_tox_slope",
            "vth_interaction_slope_V",
            "log_mu_interaction_slope",
            "log_n_minus_one_interaction_slope",
            "log_dibl_interaction_slope",
            "log_lambda_interaction_slope",
        )
        for name in slope_names:
            if not np.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite.")
        self.envelope.require_contains(self.length_ref_m, self.tox_ref_m)

    def normalized_geometry(
        self,
        length_m: float,
        tox_m: float,
        *,
        enforce_envelope: bool = True,
    ) -> tuple[float, float]:
        if (
            not np.isfinite(float(length_m))
            or not np.isfinite(float(tox_m))
            or float(length_m) <= 0.0
            or float(tox_m) <= 0.0
        ):
            raise ValueError("Device length and tox must be finite and positive.")
        if enforce_envelope:
            self.envelope.require_contains(length_m, tox_m)
        return (
            (float(length_m) - self.length_ref_m) / self.length_ref_m,
            (float(tox_m) - self.tox_ref_m) / self.tox_ref_m,
        )

    def effective_parameters(
        self,
        length_m: float,
        tox_m: float,
        *,
        device_width_m: float,
        enforce_envelope: bool = True,
    ) -> EnhancedModelParameters:
        device_width_m = float(device_width_m)
        if not np.isfinite(device_width_m) or device_width_m <= 0.0:
            raise ValueError("Device width must be finite and positive.")
        x_l, x_t = self.normalized_geometry(
            length_m,
            tox_m,
            enforce_envelope=enforce_envelope,
        )
        interaction = x_l * x_t
        vth = (
            self.vth_ref_V
            + self.vth_length_slope_V * x_l
            + self.vth_tox_slope_V * x_t
            + self.vth_interaction_slope_V * interaction
        )
        mu = self.mu_ref_m2_per_Vs * np.exp(
            self.log_mu_length_slope * x_l
            + self.log_mu_tox_slope * x_t
            + self.log_mu_interaction_slope * interaction
        )
        subthreshold_n = 1.0 + (self.subthreshold_n_ref - 1.0) * np.exp(
            self.log_n_minus_one_length_slope * x_l
            + self.log_n_minus_one_tox_slope * x_t
            + self.log_n_minus_one_interaction_slope * interaction
        )
        dibl_coeff = self.dibl_ref_V_per_V * np.exp(
            self.log_dibl_length_slope * x_l
            + self.log_dibl_tox_slope * x_t
            + self.log_dibl_interaction_slope * interaction
        )
        lambda_clm = self.lambda_ref_1_per_V * np.exp(
            self.log_lambda_length_slope * x_l
            + self.log_lambda_tox_slope * x_t
            + self.log_lambda_interaction_slope * interaction
        )
        values = np.asarray(
            [vth, mu, subthreshold_n, dibl_coeff, lambda_clm],
            dtype=float,
        )
        if not np.isfinite(values).all() or np.any(values[1:] <= 0.0):
            raise ValueError(
                "Geometry mapping produced non-finite or non-positive parameters."
            )
        return EnhancedModelParameters(
            w=device_width_m,
            l=float(length_m),
            tox=float(tox_m),
            mu=float(mu),
            vth=float(vth),
            lambda_clm=float(lambda_clm),
            subthreshold_n=float(subthreshold_n),
            temperature=self.temperature_K,
            i0=self.i0_A,
            dibl_coeff=float(dibl_coeff),
            l_ref=self.length_ref_m,
            theta_mobility=self.theta_mobility,
        )

    def ids(
        self,
        *,
        vgs: float | np.ndarray,
        vds: float | np.ndarray,
        length_m: float,
        tox_m: float,
        device_width_m: float,
        enforce_envelope: bool = True,
    ) -> float | np.ndarray:
        params = self.effective_parameters(
            length_m,
            tox_m,
            device_width_m=device_width_m,
            enforce_envelope=enforce_envelope,
        )
        ids = ids_nmos_enhanced(vgs, vds, params)
        values = np.asarray(ids, dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError("Geometry-aware model produced invalid current.")
        return ids

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
    ) -> "GeometryAwareModelParameters":
        values = dict(raw)
        envelope_raw = values.pop("envelope")
        if not isinstance(envelope_raw, Mapping):
            raise ValueError("Geometry-aware model envelope must be a mapping.")
        return cls(
            **values,
            envelope=GeometryEnvelope(**dict(envelope_raw)),
        )
