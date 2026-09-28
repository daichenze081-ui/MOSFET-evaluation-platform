from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mosfet_platform.artifacts.frozen_model import FrozenModel
from mosfet_platform.extraction.formal import (
    BundleBuildResult,
    CurveIssue,
    CurveTrace,
    DeviceCurveBundle,
    build_bundle,
    curve_rows as rows_from_curves,
)
from mosfet_platform.measurement import MeasurementContract


def _geometry(
    row: Mapping[str, Any], contract: MeasurementContract
) -> tuple[str, float, float, float]:
    device_id = str(row.get("device_id", "")).strip()
    if not device_id or device_id.lower() == "nan":
        raise ValueError("device_id must not be empty.")
    values = tuple(
        float(row[name])
        for name in ("width_m", "length_m", "oxide_thickness_m")
    )
    if not np.isfinite(values).all() or min(values) <= 0.0:
        raise ValueError("Geometry must be finite and positive.")
    width, length, tox = values
    if not np.isclose(width, contract.geometry.width_m, rtol=1.0e-9, atol=0.0):
        raise ValueError("width_m does not match the MeasurementContract.")
    return device_id, width, length, tox


def _traces(
    width: float,
    length: float,
    tox: float,
    contract: MeasurementContract,
    frozen: FrozenModel,
    enforce_envelope: bool,
) -> tuple[CurveTrace, CurveTrace, CurveTrace]:
    vgs = np.linspace(
        contract.transfer.vgs_start_V,
        contract.transfer.vgs_stop_V,
        contract.transfer.num_points,
    )
    vds = np.linspace(0.0, contract.ion.vds_V, contract.transfer.num_points)
    definitions = (
        ("idvg", contract.transfer.vds_V, vgs, np.full_like(vgs, contract.transfer.vds_V)),
        ("idvg", contract.ion.vds_V, vgs, np.full_like(vgs, contract.ion.vds_V)),
        ("idvd", contract.ion.vgs_V, np.full_like(vds, contract.ion.vgs_V), vds),
    )
    result: list[CurveTrace] = []
    for kind, bias, vgs_values, vds_values in definitions:
        current = frozen.model.ids(
            vgs=vgs_values,
            vds=vds_values,
            length_m=length,
            tox_m=tox,
            device_width_m=width,
            enforce_envelope=enforce_envelope,
        )
        result.append(
            CurveTrace(
                curve_type=kind,
                fixed_bias_V=float(bias),
                sweep=vgs_values if kind == "idvg" else vds_values,
                current_A=np.asarray(current, dtype=float),
            )
        )
    return result[0], result[1], result[2]


def build_predicted_bundle(
    row: Mapping[str, Any],
    contract: MeasurementContract,
    frozen: FrozenModel,
    *,
    diagnostics: bool = False,
) -> tuple[BundleBuildResult, dict[str, Any]]:
    provenance = {
        "result_origin": "predicted",
        "source_id": frozen.run_id,
        "model_id": frozen.run_id,
        "model_sha256": frozen.model_sha256,
        "model_family": frozen.family,
    }
    try:
        device_id, width, length, tox = _geometry(row, contract)
    except (KeyError, TypeError, ValueError) as error:
        issue = CurveIssue("INVALID_INPUT", str(error), False)
        return BundleBuildResult(None, (issue,)), provenance

    in_envelope = frozen.model.envelope.contains(length, tox)
    if not in_envelope and not diagnostics:
        issue = CurveIssue("OUT_OF_ENVELOPE", "Geometry is outside the qualified envelope.", False)
        return BundleBuildResult(None, (issue,)), provenance

    try:
        traces = _traces(width, length, tox, contract, frozen, in_envelope)
    except (ValueError, FloatingPointError, OverflowError) as error:
        issue = CurveIssue("MODEL_ERROR", str(error), False)
        return BundleBuildResult(None, (issue,)), provenance
    if not in_envelope:
        issue = CurveIssue("OUT_OF_ENVELOPE", "Geometry is outside the qualified envelope.", False)
        return BundleBuildResult(None, (issue,), traces), provenance

    bundle = build_bundle(
        device_id=device_id,
        condition_id=contract.condition_id,
        geometry=(width, length, tox),
        curves=traces,
        contract=contract,
        source="generated",
    )
    return BundleBuildResult(bundle), provenance


def curve_rows(bundle: DeviceCurveBundle, source: Mapping[str, Any]) -> list[dict[str, Any]]:
    return rows_from_curves(
        device_id=bundle.device_id,
        condition_id=bundle.condition_id,
        geometry=(bundle.width_m, bundle.length_m, bundle.oxide_thickness_m),
        curves=bundle.curves,
        meta={"envelope_status": "IN_ENVELOPE", **source},
    )


def diagnostic_rows(
    row: Mapping[str, Any],
    contract: MeasurementContract,
    curves: tuple[CurveTrace, ...],
    source: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return rows_from_curves(
        device_id=str(row["device_id"]),
        condition_id=contract.condition_id,
        geometry=tuple(float(row[name]) for name in ("width_m", "length_m", "oxide_thickness_m")),
        curves=curves,
        meta={"envelope_status": "UNQUALIFIED_EXTRAPOLATION", **source},
    )
