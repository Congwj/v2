from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
