from abc import ABC, abstractmethod
from typing import Any, Optional


class StorageBackend(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        ...

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        ...

    @abstractmethod
    def keys(self) -> list[str]:
        ...


class InMemoryStorage(StorageBackend):
    def __init__(self):
        self._store: dict[str, Any] = {}

    def get(self, key):
        return self._store.get(key)

    def set(self, key, value):
        self._store[key] = value

    def delete(self, key):
        self._store.pop(key, None)

    def keys(self):
        return list(self._store.keys())
