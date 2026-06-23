"""Durable receive store contract.

The in-memory store is for local smoke tests and dev-only operation. Production
must replace it with SQLite or My Stand's server-side durable store before
public traffic is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .events import DeliveryKey, DurableReceiveResult, NormalizedInboundEvent


@dataclass
class InMemoryDurableReceiveStore:
    seen: set[str] = field(default_factory=set)

    def accept(self, event: NormalizedInboundEvent) -> DurableReceiveResult:
        key = event.delivery_key
        stable = key.stable_key
        if stable in self.seen:
            return DurableReceiveResult(accepted=False, delivery_key=key, status="duplicate")
        self.seen.add(stable)
        return DurableReceiveResult(accepted=True, delivery_key=key, status="accepted")

    def has_seen(self, key: DeliveryKey) -> bool:
        return key.stable_key in self.seen

