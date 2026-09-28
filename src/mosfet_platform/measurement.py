from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


@dataclass(frozen=True)
class GeometryCondition:
    width_m: float
    length_m: float

    def __post_init__(self) -> None:
        if self.width_m <= 0.0 or self.length_m <= 0.0:
            raise ValueError("Geometry width_m and length_m must be positive.")

    @property
    def w_over_l(self) -> float:
        return self.width_m / self.length_m


@dataclass(frozen=True)
class IdVgCondition:
    vds_V: float
    vgs_start_V: float
    vgs_stop_V: float
    vgs_off_V: float
    vgs_on_V: float
    num_points: int

    def __post_init__(self) -> None:
        if self.vds_V < 0.0:
            raise ValueError("transfer.vds_V must be non-negative.")
        if self.vgs_stop_V <= self.vgs_start_V:
            raise ValueError("transfer.vgs_stop_V must exceed vgs_start_V.")
        if self.num_points < 3:
            raise ValueError("transfer.num_points must be at least 3.")
        for name, value in (
            ("vgs_off_V", self.vgs_off_V),
            ("vgs_on_V", self.vgs_on_V),
        ):
            if not self.vgs_start_V <= value <= self.vgs_stop_V:
                raise ValueError(f"transfer.{name} must lie inside the sweep.")


@dataclass(frozen=True)
class VthCondition:
    method: str
    reference_current_A_per_W_over_L: float

    def __post_init__(self) -> None:
        if self.method != "constant_current_w_over_l":
            raise ValueError("Vth method must be constant_current_w_over_l.")
        if self.reference_current_A_per_W_over_L <= 0.0:
            raise ValueError("Vth normalized reference current must be positive.")

    def target_current_A(self, geometry: GeometryCondition) -> float:
        return self.reference_current_A_per_W_over_L * geometry.w_over_l


@dataclass(frozen=True)
class SSCondition:
    method: str
    current_min_A: float
    current_max_A: float
    minimum_points: int = 3

    def __post_init__(self) -> None:
        if self.method != "log_linear_fit":
            raise ValueError("Only log_linear_fit SS extraction is supported.")
        if self.current_min_A <= 0.0 or self.current_max_A <= self.current_min_A:
            raise ValueError("SS current window must be positive and increasing.")
        if self.minimum_points < 3:
            raise ValueError("ss.minimum_points must be at least 3.")


@dataclass(frozen=True)
class CurrentCondition:
    metric_source: str = "magnitude"
    preserve_signed_current: bool = True
    sampling_method: str = "linear"
    bias_voltage_tolerance_V: float = 1.0e-12
    allow_interpolation: bool = True

    def __post_init__(self) -> None:
        if self.metric_source != "magnitude":
            raise ValueError("Only magnitude current is supported.")
        if self.sampling_method not in {"exact", "linear", "nearest"}:
            raise ValueError("Unsupported current sampling_method.")
        tolerance = float(self.bias_voltage_tolerance_V)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(
                "bias_voltage_tolerance_V must be finite and non-negative."
            )
        if self.sampling_method == "exact" and self.allow_interpolation:
            raise ValueError("Exact sampling cannot allow interpolation.")
        if self.sampling_method == "linear" and not self.allow_interpolation:
            raise ValueError("Linear sampling requires interpolation.")


@dataclass(frozen=True)
class IonCondition:
    """Formal Ion bias and cross-curve consistency contract."""

    vgs_V: float
    vds_V: float
    primary_curve_type: str
    verification_curve_type: str
    consistency_relative_tolerance: float
    consistency_absolute_tolerance_A: float

    def __post_init__(self) -> None:
        if self.vgs_V < 0.0 or self.vds_V < 0.0:
            raise ValueError("Ion Vgs and Vds must be non-negative.")
        if self.primary_curve_type != "idvg":
            raise ValueError("Formal Ion primary_curve_type must be 'idvg'.")
        if self.verification_curve_type != "idvd":
            raise ValueError("Formal Ion verification_curve_type must be 'idvd'.")
        if not 0.0 < self.consistency_relative_tolerance < 1.0:
            raise ValueError("Ion relative consistency tolerance must lie in (0, 1).")
        if self.consistency_absolute_tolerance_A <= 0.0:
            raise ValueError("Ion absolute consistency tolerance must be positive.")


@dataclass(frozen=True)
class DiblCondition:
    """Exact Id-Vg Vds pair used for DIBL extraction and validation."""

    low_vds_V: float
    high_vds_V: float

    def __post_init__(self) -> None:
        if self.low_vds_V < 0.0:
            raise ValueError("DIBL low_vds_V must be non-negative.")
        if self.high_vds_V <= self.low_vds_V:
            raise ValueError("DIBL high_vds_V must exceed low_vds_V.")


@dataclass(frozen=True)
class MeasurementContract:
    """Single source of truth for formal multi-bias MOSFET metrics."""

    condition_id: str
    device_type: str
    temperature_K: float
    geometry: GeometryCondition
    transfer: IdVgCondition
    ion: IonCondition
    dibl: DiblCondition
    vth: VthCondition
    ss: SSCondition
    current: CurrentCondition

    def __post_init__(self) -> None:
        if not self.condition_id.strip():
            raise ValueError("condition_id must not be empty.")
        if self.device_type.lower() not in {"nmos", "pmos"}:
            raise ValueError("device_type must be 'nmos' or 'pmos'.")
        if self.temperature_K <= 0.0:
            raise ValueError("temperature_K must be positive.")
        if not np.isclose(
            self.transfer.vds_V,
            self.dibl.low_vds_V,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("Transfer Vds must equal the DIBL low-Vds bias.")
        if not np.isclose(
            self.ion.vds_V,
            self.dibl.high_vds_V,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("Formal Ion Vds must equal the DIBL high-Vds bias.")
        if not self.transfer.vgs_start_V <= self.ion.vgs_V <= self.transfer.vgs_stop_V:
            raise ValueError("Formal Ion Vgs must lie inside the Id-Vg sweep.")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "MeasurementContract":
        section = raw.get("measurement_contract")
        if not isinstance(section, Mapping):
            raise ValueError("measurement_contract must be a mapping.")
        geometry_raw = section.get("geometry")
        if not isinstance(geometry_raw, Mapping):
            raise ValueError("measurement_contract.geometry is required.")
        transfer_raw = section.get("transfer")
        if not isinstance(transfer_raw, Mapping):
            raise ValueError("measurement_contract.transfer is required.")
        ion_raw = section.get("ion")
        if not isinstance(ion_raw, Mapping):
            raise ValueError("measurement_contract.ion is required.")
        dibl_raw = section.get("dibl")
        if not isinstance(dibl_raw, Mapping):
            raise ValueError("measurement_contract.dibl is required.")
        current_raw = section.get("current")
        if not isinstance(current_raw, Mapping):
            raise ValueError("measurement_contract.current is required.")
        required_sampling_fields = {
            "sampling_method",
            "bias_voltage_tolerance_V",
            "allow_interpolation",
        }
        missing_sampling_fields = sorted(required_sampling_fields - set(current_raw))
        if missing_sampling_fields:
            raise ValueError(
                "Formal measurement_contract.current is missing sampling policy fields: "
                + ", ".join(missing_sampling_fields)
            )
        return cls(
            condition_id=str(section["condition_id"]),
            device_type=str(section.get("device_type", "nmos")),
            temperature_K=float(section["temperature_K"]),
            geometry=GeometryCondition(**dict(geometry_raw)),
            transfer=IdVgCondition(**dict(transfer_raw)),
            ion=IonCondition(**dict(ion_raw)),
            dibl=DiblCondition(**dict(dibl_raw)),
            vth=VthCondition(**dict(section["vth"])),
            ss=SSCondition(**dict(section["ss"])),
            current=CurrentCondition(**dict(current_raw)),
        )

    @property
    def vth_target_current_A(self) -> float:
        return self.vth.target_current_A(self.geometry)

    def target_current_A(self, *, width_m: float, length_m: float) -> float:
        return self.vth.target_current_A(
            GeometryCondition(width_m=float(width_m), length_m=float(length_m))
        )

    def metric_bias(self, metric: str) -> tuple[float | None, float]:
        if metric in {"ion", "ion_ioff_cross_bias"}:
            return self.ion.vgs_V, self.ion.vds_V
        if metric == "ioff":
            return self.transfer.vgs_off_V, self.transfer.vds_V
        if metric in {"same_vds_on_current", "ion_ioff_same_vds_0p1"}:
            return self.transfer.vgs_on_V, self.transfer.vds_V
        if metric in {"vth", "ss_mv_dec", "ss_v_dec", "gm_max"}:
            return None, self.transfer.vds_V
        raise ValueError(f"Unknown formal metric: {metric}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate_model_context(
        self,
        *,
        width_m: float,
        length_m: float,
        temperature_K: float,
    ) -> None:
        checks = (
            ("width", width_m, self.geometry.width_m),
            ("length", length_m, self.geometry.length_m),
            ("temperature", temperature_K, self.temperature_K),
        )
        for name, actual, expected in checks:
            if not np.isclose(float(actual), float(expected), rtol=1.0e-9, atol=0.0):
                raise ValueError(f"Model {name} does not match MeasurementContract.")


def load_measurement_contract(path: str | Path) -> MeasurementContract:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Measurement contract config not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, Mapping):
        raise ValueError("Measurement contract YAML must contain a mapping.")
    return MeasurementContract.from_mapping(raw)
