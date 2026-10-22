"""Tracks ingested documents to prevent redundant parsing and maintain state."""

from pathlib import Path
from typing import Dict, Optional, Union
from src.domain.documents import Document
from src.utils.file_utils import compute_file_hash, load_json, save_json


class DocumentRegistry:
    """Maintains a JSON record of indexed documents and their checksum hashes."""

    def __init__(self, registry_path: Union[str, Path]):
        self.registry_path = Path(registry_path)
        self.registry: Dict[str, Dict] = self._load()

    def _load(self) -> Dict[str, Dict]:
        if self.registry_path.exists():
            try:
                return load_json(self.registry_path)
            except Exception:
                return {}
        return {}

    def save(self) -> None:
        save_json(self.registry, self.registry_path)

    def is_document_registered(self, file_path: Union[str, Path]) -> bool:
        """Check if file hash already exists in registry without modification."""
        path = Path(file_path)
        if not path.is_file():
            return False

        current_hash = compute_file_hash(path)
        for record in self.registry.values():
            if record.get("file_hash") == current_hash:
                return True
        return False

    def register(self, document: Document) -> None:
        """Add or update a document in the registry."""
        self.registry[document.doc_id] = {
            "doc_id": document.doc_id,
            "file_name": document.metadata.file_name,
            "source_path": document.metadata.source_path,
            "file_hash": document.metadata.file_hash,
            "total_pages": document.metadata.total_pages,
            "title": document.metadata.title,
            "created_at": document.metadata.created_at.isoformat(),
        }
        self.save()

    def get_document_info(self, doc_id: str) -> Optional[Dict]:
        return self.registry.get(doc_id)

    def get_all_documents(self) -> list[Dict]:
        return list(self.registry.values())