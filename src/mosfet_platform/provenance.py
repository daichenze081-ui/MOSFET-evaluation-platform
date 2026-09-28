from __future__ import annotations

from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import subprocess
import tomllib


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_relative_path(path: str | Path, project_root: str | Path = ".") -> str:
    root = Path(project_root).resolve()
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return Path(path).name


def git_commit(project_root: str | Path = ".") -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(project_root),
            capture_output=True, text=True, check=True, timeout=5,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def software_version() -> str:
    pyproject_path = Path("pyproject.toml")
    if pyproject_path.exists():
        with pyproject_path.open("rb") as file:
            project = tomllib.load(file).get("project", {})
        if project.get("version"):
            return str(project["version"])
    try:
        return version("mosfet-test-platform")
    except PackageNotFoundError:
        return "0.2.0"


def manifest_file_entry(path: str | Path, project_root: str | Path = ".") -> dict[str, str]:
    return {
        "path": project_relative_path(path, project_root),
        "sha256": file_sha256(path),
    }
