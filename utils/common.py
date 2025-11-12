import os
import sys
from pathlib import Path
from typing import Any, Dict


def coerce_to_dict(value: Any) -> Dict[str, Any]:
    """Best-effort conversion to a dict."""
    if isinstance(value, dict):
        return value
    try:
        return dict(value)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001 - conversion is best effort
        return {}


def load_local_env(module_path: str, *, filename: str = ".env") -> None:
    """Load environment variables from a sibling .env file if present."""
    env_path = Path(module_path).with_name(filename)
    if not env_path.exists():
        return

    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            cleaned_value = value.strip().strip('"').strip("'")
            os.environ[key] = cleaned_value
    except OSError as exc:
        print(f"Warning: failed to read {env_path.name}: {exc}", file=sys.stderr)
