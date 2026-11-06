"""Close native Chroma handles before temporary directories are removed."""
from pathlib import Path


def close_chroma(directory):
    from chromadb.api.shared_system_client import SharedSystemClient
    root = Path(directory).resolve()
    for identifier, system in list(SharedSystemClient._identifier_to_system.items()):
        if Path(identifier).resolve().is_relative_to(root):
            system.stop()
            SharedSystemClient._identifier_to_system.pop(identifier, None)
