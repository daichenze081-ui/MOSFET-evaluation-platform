"""Stable local service boundary for CLI clients and future agent tools."""

from __future__ import annotations

from collections.abc import Mapping
from uuid import uuid4

import yaml

from mosfet_platform.analysis.diagnosis import GROUP_FIELDS
from mosfet_platform.analysis.completion import validate_completion
from mosfet_platform.analysis.spec import SPEC_RULES
from mosfet_platform.workflows.diagnose import run_diagnosis

DIAGNOSIS_REQUEST_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["contract", "spec"],
    "properties": {
        **{key: {"type": "string", "minLength": 1}
           for key in ("contract", "spec", "cases", "metrics", "root", "output", "device_id")},
        "source_type": {"type": "string", "enum": ["measured", "comsol", "synthetic", "estimated"]},
        "group_by": {"type": "array", "items": {"type": "string", "enum": list(GROUP_FIELDS)}, "uniqueItems": True},
        "completion": {"type": "object", "additionalProperties": False, "required": ["model_manifest"],
                       "properties": {"model_manifest": {"type": "string", "minLength": 1},
                                      "absolute_tolerances": {"type": "object", "additionalProperties": False,
                                                              "properties": {rule[1]: {"type": "number", "minimum": 0} for rule in SPEC_RULES}}}},
    },
    "oneOf": [{"required": ["cases"], "not": {"required": ["metrics"]}},
              {"required": ["metrics"], "not": {"required": ["cases"]}}],
}


def diagnose(request: Mapping) -> dict:
    """Execute one diagnosis. Device FAIL is data, not an execution error."""
    try:
        if not isinstance(request, Mapping):
            raise ValueError("Diagnostic request must be an object.")
        allowed = DIAGNOSIS_REQUEST_SCHEMA["properties"]
        if set(request) - set(allowed):
            raise ValueError("Unknown request fields: " + ", ".join(sorted(set(request) - set(allowed))))
        if any(key not in request for key in ("contract", "spec")):
            raise ValueError("contract and spec are required.")
        for key, value in request.items():
            if key == "completion":
                validate_completion(value)
            elif key == "group_by":
                if not isinstance(value, list) or any(not isinstance(v, str) or v not in GROUP_FIELDS for v in value):
                    raise ValueError("group_by must be an array of supported field names.")
                if len(value) != len(set(value)):
                    raise ValueError("group_by must not contain duplicates.")
            elif not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be a non-empty string.")
        if "source_type" in request and request["source_type"] not in allowed["source_type"]["enum"]:
            raise ValueError("Unsupported source_type.")
        return run_diagnosis(**dict(request))
    except (ValueError, OSError, RuntimeError, KeyError, TypeError, yaml.YAMLError) as error:
        return {"schema_version": "1.0", "run_id": uuid4().hex, "status": "FAILED",
                "error": {"code": "INPUT_NOT_FOUND" if isinstance(error, FileNotFoundError)
                          else "INVALID_REQUEST" if isinstance(error, (ValueError, KeyError, TypeError, yaml.YAMLError))
                          else "EXECUTION_FAILED", "message": str(error)}, "artifacts": {}}
