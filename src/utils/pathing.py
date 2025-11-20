from pathlib import Path
from typing import Optional

def find_project_root(start: Optional[Path] = None) -> Path:
    p = Path.cwd() if start is None else Path(start)
    for ancestor in [p, *p.parents]:
        if (ancestor / 'pyproject.toml').exists() or (ancestor / '.git').exists() or (ancestor / 'src').exists():
            return ancestor
    return p

def project_paths():
    root = find_project_root()
    return {
        "root": root,
        "src": root / "src",
        "config": root / "config",
        "data": (root / "ds002680"),
        "notebooks": root / "notebooks",
    }

def ensure_src_on_path():
    import sys
    paths = project_paths()
    src = str(paths["src"])
    if src not in sys.path:
        sys.path.insert(0, src)
