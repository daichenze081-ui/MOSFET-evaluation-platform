"""Load diagnostic observations without requiring a complete training bundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mosfet_platform.analysis.diagnosis import canonical_source
from mosfet_platform.extraction.metrics import (
    extract_gm_max, extract_ion_ioff_ratio, extract_ion_sample_at_bias,
    extract_subthreshold_swing_robust, extract_vth_constant_current_robust,
    require_ion_consistency, sample_current_at_bias,
)
from mosfet_platform.io.measured import ManifestMeasuredSource, build_measured_bundle
from mosfet_platform.io.metric_table import _context_errors, read_metric_table


def metric_records(path, contract, *, source_type=None):
    records = []
    declared = canonical_source(source_type) if source_type else None
    for row in read_metric_table(path, unique_device=False).to_dict(orient="records"):
        declarations = []
        for key in ("source_type", "upstream_source_type"):
            value = str(row.get(key, "")).strip().lower()
            if value and value not in {"metric_table", "external"}:
                declarations.append(canonical_source(value))
        for key in ("result_origin", "upstream_result_origin"):
            if str(row.get(key, "")).lower().startswith("predict"):
                declarations.append("estimated")
        if declared:
            declarations.append(declared)
        if not declarations:
            raise ValueError("Metric input needs source_type in its rows or the diagnostic request.")
        if len(set(declarations)) != 1:
            raise ValueError(f"Conflicting source declarations for device {row['device_id']}.")
        # Metric-value faults are evaluated separately; one negative value must not hide other metrics.
        errors = [e for e in _context_errors(row, contract) if not e.startswith("negative_metric:")]
        records.append({**row, "source_type": declarations[0], "context_errors": errors,
                        "method": "imported_metrics", "evidence": [str(path.resolve())]})
    return records


def curve_records(manifest, contract, root: Path, *, source_type=None):
    source = canonical_source(manifest.raw["source"]["type"])
    if source_type and canonical_source(source_type) != source:
        raise ValueError("Request source_type conflicts with the curve manifest.")
    records = []
    for raw in ManifestMeasuredSource(manifest).load_batch():
        built, _ = build_measured_bundle(raw, contract, root, diagnostics=True)
        traces = built.bundle.curves if built.bundle else built.diagnostic_curves
        row = {"device_id": raw.device_id, "condition_id": contract.condition_id,
               "device_type": contract.device_type, "temperature_K": contract.temperature_K,
               "width_m": raw.width_m, **raw.case["geometry"], "source_type": source,
               "method": "curve_extraction", "metric_issues": {}, "metric_evidence": {},
               "issues": [{"code": issue.code, "message": issue.message} for issue in built.issues]}
        if "batch_id" in raw.case:
            row["batch_id"] = raw.case["batch_id"]
        tol = contract.current.bias_voltage_tolerance_V
        roles = {"transfer": ("idvg", contract.transfer.vds_V),
                 "ion": ("idvg", contract.ion.vds_V), "verification": ("idvd", contract.ion.vgs_V)}
        chosen, files, absent = {}, {}, {}
        for role, (kind, bias) in roles.items():
            candidates = [t for t in traces if t.curve_type == kind
                          and np.isclose(t.fixed_bias_V, bias, rtol=0, atol=tol)]
            chosen[role] = candidates[0] if len(candidates) == 1 else None
            bias_key = "vds_V" if kind == "idvg" else "vgs_V"
            entries = [c for c in raw.case[kind] if c["qc_status"] == "active"
                       and c["analysis_role"] == "formal"
                       and np.isclose(float(c[bias_key]), bias, rtol=0, atol=tol)]
            files[role] = [str((root / c["path"]).resolve()) for c in entries]
            absent[role] = "MISSING" if not entries else "INVALID"

        def extract(metric, dependencies, calculate):
            row["metric_evidence"][metric] = list(dict.fromkeys(
                p for dependency in dependencies for p in files[dependency]))
            missing = [name for name in dependencies if chosen[name] is None]
            if missing:
                state = "INVALID" if any(absent[name] == "INVALID" for name in missing) else "MISSING"
                row["metric_issues"][metric] = {"status": state, "reason": "unavailable_curve:" + ",".join(missing)}
                return
            try:
                value = calculate()
                if not np.isfinite(value):
                    raise ValueError("Extraction produced a non-finite value.")
                row[metric] = float(value)
            except (ValueError, FloatingPointError, OverflowError) as error:
                row["metric_issues"][metric] = {"status": "INVALID", "reason": str(error)}

        def current(role, vgs, vds):
            trace = chosen[role]
            return extract_ion_sample_at_bias(
                sweep_values=trace.sweep, ids=trace.current_A, curve_type=trace.curve_type,
                fixed_bias_V=trace.fixed_bias_V, target_vgs_V=vgs, target_vds_V=vds,
                measurement_contract=contract,
            ).current_A

        def ioff():
            trace = chosen["transfer"]
            return sample_current_at_bias(
                trace.sweep, trace.current_A, target_voltage_V=contract.transfer.vgs_off_V,
                method=contract.current.sampling_method,
                voltage_tolerance_V=tol, allow_interpolation=contract.current.allow_interpolation,
            ).current_A

        def vth():
            trace = chosen["transfer"]
            value, status = extract_vth_constant_current_robust(
                trace.sweep, trace.current_A,
                target_current=contract.target_current_A(width_m=raw.width_m, length_m=row["length_m"]),
            )
            row["vth_status"] = status
            return value

        def ss():
            trace = chosen["transfer"]
            value, status = extract_subthreshold_swing_robust(
                trace.sweep, trace.current_A, current_min=contract.ss.current_min_A,
                current_max=contract.ss.current_max_A, minimum_points=contract.ss.minimum_points,
            )
            row["ss_status"] = status
            return value * 1000.0

        def ion():
            primary = current("ion", contract.ion.vgs_V, contract.ion.vds_V)
            verification = current("verification", contract.ion.vgs_V, contract.ion.vds_V)
            require_ion_consistency(primary, verification, measurement_contract=contract)
            return primary

        extract("ioff", ("transfer",), ioff)
        extract("vth", ("transfer",), vth)
        extract("ss_mv_dec", ("transfer",), ss)
        extract("gm_max", ("transfer",), lambda: extract_gm_max(chosen["transfer"].sweep, chosen["transfer"].current_A))
        extract("ion", ("ion", "verification"), ion)
        extract("ion_ioff", ("transfer",), lambda: extract_ion_ioff_ratio(
            current("transfer", contract.transfer.vgs_on_V, contract.transfer.vds_V), ioff()))
        extract("ion_ioff_cross_bias", ("transfer", "ion", "verification"),
                lambda: extract_ion_ioff_ratio(ion(), ioff()))
        records.append(row)
    return records
