import hashlib


def hash_segment(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
