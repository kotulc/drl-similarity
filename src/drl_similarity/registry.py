"""Generic string-keyed factory registry."""

from __future__ import annotations

from typing import Generic, TypeVar

from drl_similarity.errors import RegistryError


T = TypeVar("T")


class Registry(Generic[T]):
    """Explicit dictionary from stable YAML IDs to factories or objects."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, T] = {}

    def register(self, item_id: str, factory: T) -> T:
        if item_id in self._items:
            raise RegistryError(
                f"Duplicate {self.name} id {item_id!r}",
                context={"registry": self.name, "id": item_id},
            )
        self._items[item_id] = factory
        return factory

    def get(self, item_id: str, *, field: str = "id") -> T:
        try:
            return self._items[item_id]
        except KeyError as exc:
            valid = ", ".join(self.ids()) or "(none registered)"
            raise RegistryError(
                f"Unknown {self.name} id {item_id!r}; valid ids: {valid}",
                context={"registry": self.name, "field": field, "id": item_id},
            ) from exc

    def ids(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, item_id: object) -> bool:
        return isinstance(item_id, str) and item_id in self._items

