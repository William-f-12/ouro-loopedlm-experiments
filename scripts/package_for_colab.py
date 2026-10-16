"""Build a small, deterministic upload ZIP; requires only standard Python."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = "ouro-loopedlm-experiments"
ROOT_FILES = (
    "inference.py", "benchmark.py", "requirements.txt", "requirements-bench.txt", "README.md", "LICENSE",
    "UPSTREAM.md", "upstream.json", ".gitignore", ".gitattributes",
    "baseline_snapshot.json", "RQ2_README.md",
)
SOURCE_DIRS = ("ouro_local", "experiments", "scripts", "tests", "colab", "docs")
SOURCE_SUFFIXES = {".py", ".json", ".ipynb", ".md", ".txt"}
REQUIRED = (
    "inference.py", "requirements.txt", "ouro_local/__init__.py",
    "ouro_local/modeling_ouro.py", "ouro_local/configuration_ouro.py",
    "ouro_local/config.json",
)


def create_bundle(project_root: Path = ROOT, output_dir: Path | None = None) -> Path:
    root = Path(project_root).resolve()
    for name in REQUIRED:
        if not (root / name).is_file():
            raise ValueError(f"Missing project file: {name}")
    paths = [root / name for name in ROOT_FILES if (root / name).is_file()]
    for dirname in SOURCE_DIRS:
        for path in (root / dirname).rglob("*"):
            relative = path.relative_to(root)
            if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
                continue
            if path.is_file() and path.suffix in SOURCE_SUFFIXES:
                paths.append(path)
    payload = {}
    for path in sorted(paths):
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Source links outside this project: {path}")
        data = path.read_bytes()
        if path.suffix == ".ipynb":
            # A notebook saved after use must not upload saved prompt/output data.
            notebook = json.loads(data)
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    cell["outputs"] = []
                    cell["execution_count"] = None
            data = (json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
        payload[path.relative_to(root).as_posix()] = data
    files = {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
             for name, data in sorted(payload.items())}
    manifest_bytes = (json.dumps({"format": 1, "files": files}, sort_keys=True, indent=2) + "\n").encode()
    bundle_id = hashlib.sha256(manifest_bytes).hexdigest()[:12]
    target_dir = Path(output_dir) if output_dir is not None else root / "dist"
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / f"{ARCHIVE_ROOT}-{bundle_id}.zip"
    payload["bundle_manifest.json"] = manifest_bytes
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(payload.items()):
            entry = zipfile.ZipInfo(f"{ARCHIVE_ROOT}/{name}", date_time=(2020, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, data)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Default: project/dist")
    args = parser.parse_args()
    path = create_bundle(output_dir=args.output_dir)
    print(f"Upload this ZIP in Colab:\n{path}")
    print(f"Size: {path.stat().st_size / 1024:.1f} KiB (no model weights)")


if __name__ == "__main__":
    main()
