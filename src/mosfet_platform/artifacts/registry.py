"""Transactional local storage for measured observations and active model versions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def content_hash(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def connect_registry(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=120, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY,
            contract_hash TEXT NOT NULL,
            device_id TEXT NOT NULL,
            kind TEXT NOT NULL CHECK(kind IN ('curves','metrics')),
            role TEXT NOT NULL,
            payload_hash TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            case_json TEXT NOT NULL,
            spec_json TEXT NOT NULL,
            measured_status TEXT NOT NULL CHECK(measured_status IN ('PASS','FAIL')),
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(contract_hash, device_id, kind, payload_hash)
        );
        CREATE TABLE IF NOT EXISTS curves (
            observation_id INTEGER NOT NULL REFERENCES observations(id),
            curve_type TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            data BLOB NOT NULL
        );
        CREATE TABLE IF NOT EXISTS heads (
            contract_hash TEXT NOT NULL, device_id TEXT NOT NULL, kind TEXT NOT NULL,
            observation_id INTEGER NOT NULL REFERENCES observations(id),
            PRIMARY KEY(contract_hash, device_id, kind)
        );
        CREATE TABLE IF NOT EXISTS models (
            contract_hash TEXT PRIMARY KEY, fingerprint TEXT NOT NULL,
            version TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY, contract_hash TEXT NOT NULL,
            training_status TEXT NOT NULL, details_json TEXT NOT NULL,
            report_html TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
    """)
    return connection


def current_observations(connection: sqlite3.Connection, contract_hash: str) -> list[sqlite3.Row]:
    return connection.execute("""
        SELECT o.* FROM observations o JOIN heads h ON h.observation_id=o.id
        WHERE h.contract_hash=? ORDER BY o.id
    """, (contract_hash,)).fetchall()


def admit_observation(connection, *, contract_hash, device_id, kind, role, metrics, case, spec, status, curves):
    payload = {"case": case, "curves": [
        {"type": kind, "metadata": metadata, "sha256": hashlib.sha256(data).hexdigest()}
        for kind, metadata, data in curves
    ]}
    if kind == "metrics":
        payload["metrics"] = {key: value for key, value in metrics.items() if key not in {
            "device_id", "source_id", "source_files", "source_sha256", "source_type", "result_origin",
        }}
    digest = content_hash(payload)
    previous = connection.execute("""
        SELECT o.role FROM heads h JOIN observations o ON h.observation_id=o.id
        WHERE h.contract_hash=? AND h.device_id=? AND h.kind=?
    """, (contract_hash, device_id, kind)).fetchone()
    if previous and previous["role"] != role:
        raise ValueError(f"Dataset role cannot change for existing device {device_id}.")
    duplicate = connection.execute("""
        SELECT id FROM observations WHERE contract_hash=? AND kind=? AND payload_hash=?
    """, (contract_hash, kind, digest)).fetchone()
    if duplicate:
        return False
    cursor = connection.execute("""
        INSERT INTO observations
        (contract_hash,device_id,kind,role,payload_hash,metrics_json,case_json,spec_json,measured_status)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (contract_hash, device_id, kind, role, digest, canonical_json(metrics),
          canonical_json(case), canonical_json(spec), status))
    observation_id = cursor.lastrowid
    connection.executemany("INSERT INTO curves VALUES (?,?,?,?,?)", [
        (observation_id, curve_type, canonical_json(metadata), hashlib.sha256(data).hexdigest(), data)
        for curve_type, metadata, data in curves
    ])
    connection.execute("""
        INSERT INTO heads VALUES (?,?,?,?)
        ON CONFLICT(contract_hash,device_id,kind) DO UPDATE SET observation_id=excluded.observation_id
    """, (contract_hash, device_id, kind, observation_id))
    return True
