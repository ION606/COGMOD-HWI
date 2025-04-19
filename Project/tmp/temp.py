from pathlib import Path, PurePosixPath
import zipfile, fnmatch

ROOT = Path.cwd()
zip_path = ROOT / "report_overleaf.zip"

exclude = ['*.aux', '*.log', '*.out', '*.pdf', '*.zip', '*.pyc',
           '__pycache__', '.git*', '*.DS_Store']

def include(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    return not any(fnmatch.fnmatch(rel.as_posix(), pat) for pat in exclude)

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in ROOT.rglob('*'):
        if p.is_file() and include(p):
            zf.write(p, arcname=PurePosixPath(p.relative_to(ROOT)))

print("archive saved to", zip_path)
