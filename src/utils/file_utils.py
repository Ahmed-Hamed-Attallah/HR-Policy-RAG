"""File system and I/O utility helpers."""

import hashlib
import json
from pathlib import Path
from typing import Any, List, Union


def load_json(filepath: Union[str, Path]) -> Any:
    """Load JSON content from a file."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Alias
read_json = load_json

def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure a directory exists; creates it if it doesn't."""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def compute_file_hash(filepath: Union[str, Path], chunk_size: int = 65536) -> str:
    """Compute the SHA-256 hash of a file for change detection and deduplication."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_json(data: Any, filepath: Union[str, Path], indent: int = 2) -> None:
    """Save serializable data to a JSON file."""
    path = Path(filepath)
    ensure_directory(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(filepath: Union[str, Path]) -> Any:
    """Load JSON content from a file."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_files_by_extension(
    directory: Union[str, Path], extensions: List[str]
) -> List[Path]:
    """Recursively list all files in a directory matching specific extensions."""
    dir_path = Path(directory)
    if not dir_path.exists():
        return []

    normalized_exts = {ext.lower().lstrip(".") for ext in extensions}
    return [
        p
        for p in dir_path.rglob("*")
        if p.is_file() and p.suffix.lower().lstrip(".") in normalized_exts
    ]