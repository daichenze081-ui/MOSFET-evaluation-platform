from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from mosfet_platform.case_manifest import COMSOLCaseManifest
from mosfet_platform.extraction.formal import (
    BundleBuildResult,
    CurveIssue,
    CurveTrace,
    build_bundle,
)
from mosfet_platform.io.comsol_curve import load_comsol_curve
from mosfet_platform.measurement import MeasurementContract


@dataclass(frozen=True)
class RawMeasuredDevice:
    case: Mapping[str, Any]
    width_m: float
    source_type: str

    @property
    def device_id(self) -> str:
        return str(self.case["case_id"])


class MeasuredDataSource(Protocol):
    def load_device(self, device_id: str) -> RawMeasuredDevice: ...

    def load_batch(self, batch_id: str | None = None) -> Sequence[RawMeasuredDevice]: ...


class ManifestMeasuredSource:
    def __init__(self, manifest: COMSOLCaseManifest) -> None:
        self.manifest = manifest
        common = manifest.raw["common_conditions"]
        self.width_m = float(common["width_m"])
        self.source_type = str(manifest.raw["source"]["type"]).lower()
        self.devices = tuple(
            RawMeasuredDevice(case, self.width_m, self.source_type)
            for case in (*manifest.cases, *manifest.independent_validation_cases)
        )

    def load_device(self, device_id: str) -> RawMeasuredDevice:
        found = [device for device in self.devices if device.device_id == device_id]
        if len(found) != 1:
            raise KeyError(f"Measured device not found: {device_id}")
        return found[0]

    def load_batch(self, batch_id: str | None = None) -> Sequence[RawMeasuredDevice]:
        if batch_id not in {None, "all"}:
            raise KeyError(f"Manifest source has no batch: {batch_id}")
        return self.devices


def _select(
    case: Mapping[str, Any],
    kind: str,
    bias_key: str,
    bias: float,
    tol: float,
    code: str,
) -> tuple[Mapping[str, Any] | None, CurveIssue | None]:
    found = [
        curve
        for curve in case[kind]
        if curve["qc_status"] == "active"
        and curve["analysis_role"] == "formal"
        and np.isclose(float(curve[bias_key]), bias, rtol=0.0, atol=tol)
    ]
    if len(found) == 1:
        return found[0], None
    if not found:
        return None, CurveIssue(code, f"Required {kind} curve is missing.", True)
    return None, CurveIssue(f"DUPLICATE_{code}", f"Required {kind} curve is duplicated.", False)


def build_measured_bundle(
    raw: RawMeasuredDevice,
    contract: MeasurementContract,
    root: Path,
    *,
    diagnostics: bool = False,
) -> tuple[BundleBuildResult, dict[str, Any]]:
    case = raw.case
    tol = contract.current.bias_voltage_tolerance_V
    requests = (
        ("idvg", "vds_V", contract.transfer.vds_V, "MISSING_TRANSFER_IDVG"),
        ("idvg", "vds_V", contract.ion.vds_V, "MISSING_ION_IDVG"),
        ("idvd", "vgs_V", contract.ion.vgs_V, "MISSING_ION_IDVD"),
    )
    traces: list[CurveTrace] = []
    files: list[str] = []
    issues: list[CurveIssue] = []
    sign_available = True
    for kind, bias_key, bias, code in requests:
        curve, issue = _select(case, kind, bias_key, bias, tol, code)
        if curve is None:
            issues.append(
                issue or CurveIssue(code, f"Required {kind} curve is missing.", True)
            )
            continue
        path = root / str(curve["path"])
        try:
            frame, provenance = load_comsol_curve(
                path,
                curve_type=kind,
                fixed_bias_V=float(curve[bias_key]),
                numerical_zero_current_A=1.0e-14,
            )
            sweep = frame["vgs" if kind == "idvg" else "vds"].to_numpy(dtype=float)
            traces.append(
                CurveTrace(
                    curve_type=kind,
                    fixed_bias_V=float(curve[bias_key]),
                    sweep=sweep,
                    current_A=frame["id_magnitude"].to_numpy(dtype=float),
                )
            )
            files.append(str(curve["path"]))
            sign_available &= bool(provenance["current_sign_available"])
        except OSError as error:
            issues.append(CurveIssue("SOURCE_UNAVAILABLE", str(error), True))
        except (ValueError, FloatingPointError) as error:
            issues.append(CurveIssue("INVALID_CURVE", str(error), False))

    provenance = {
        "result_origin": "measured",
        "source_id": raw.device_id,
        "source_type": raw.source_type,
        "source_files": ";".join(files),
        "current_sign_available": sign_available,
    }
    if issues:
        return (
            BundleBuildResult(
                bundle=None,
                issues=tuple(issues),
                diagnostic_curves=tuple(traces) if diagnostics else (),
            ),
            provenance,
        )

    geometry = case["geometry"]
    try:
        bundle = build_bundle(
            device_id=raw.device_id,
            condition_id=contract.condition_id,
            geometry=(
                raw.width_m,
                float(geometry["length_m"]),
                float(geometry["oxide_thickness_m"]),
            ),
            curves=traces,
            contract=contract,
            source="manifest_validated",
        )
    except ValueError as error:
        issue = CurveIssue("INVALID_BUNDLE", str(error), False)
        return BundleBuildResult(None, (issue,), tuple(traces) if diagnostics else ()), provenance
    return BundleBuildResult(bundle), provenance
