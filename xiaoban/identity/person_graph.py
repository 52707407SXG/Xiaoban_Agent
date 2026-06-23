"""Company/person graph interface for Xiaoban-Agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class PersonProfile:
    site_id: str
    user_id: str
    display_name: str
    role: str
    title: str | None = None
    team_id: str | None = None
    manager_user_id: str | None = None
    business_tags: tuple[str, ...] = ()
    communication_notes: tuple[str, ...] = ()


class PersonGraphProvider(Protocol):
    def get_person(self, site_id: str, user_id: str) -> PersonProfile | None:
        ...


@dataclass
class InMemoryPersonGraph:
    people: dict[tuple[str, str], PersonProfile] = field(default_factory=dict)

    def add_person(self, profile: PersonProfile) -> None:
        self.people[(profile.site_id, profile.user_id)] = profile

    def get_person(self, site_id: str, user_id: str) -> PersonProfile | None:
        return self.people.get((site_id, user_id))

