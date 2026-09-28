from pathlib import Path
import subprocess


PRIVATE_PATH_FRAGMENTS = (
    "comsol" + "/exported_curves",
    "configs" + "/comsol_cases.yaml",
    "artifacts" + "/frozen_models",
    "model_generalization" + "_2026",
)


def test_public_tree_has_no_private_path_references():
    root = Path(__file__).resolve().parents[1]
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root, check=True, capture_output=True,
    ).stdout.decode("utf-8").split("\0")
    files = {
        root / name for name in listed if name and (root / name).is_file()
        and (root / name).resolve() != Path(__file__).resolve()
    }
    text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in files
    ).lower()
    for fragment in PRIVATE_PATH_FRAGMENTS:
        assert fragment.lower() not in text
