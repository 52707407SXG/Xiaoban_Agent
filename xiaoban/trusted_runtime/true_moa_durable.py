"""Plaintext-free durable idempotency and usage receipts for true MoA.

Public imports remain stable while storage ownership, monotonic accounting, and
sealed-result lifecycle are implemented in focused modules.
"""

from xiaoban.trusted_runtime.true_moa_durable_accounting import (
    _TrueMoAAccountingMixin,
)
from xiaoban.trusted_runtime.true_moa_durable_base import _TrueMoADurableBase
from xiaoban.trusted_runtime.true_moa_durable_outcomes import (
    _TrueMoAOutcomeMixin,
)
from xiaoban.trusted_runtime.true_moa_durable_shared import (
    TRUE_MOA_COMPLETED_OUTCOME_SCHEMA,
    TRUE_MOA_DURABLE_MAX_CALLS,
    TRUE_MOA_DURABLE_MAX_FINAL_CALLS,
    TRUE_MOA_DURABLE_MAX_ROWS,
    TRUE_MOA_DURABLE_USAGE_MAX_BYTES,
    TRUE_MOA_OUTCOME_BINDING_SCHEMA,
    TRUE_MOA_OUTCOME_DEFAULT_TTL_SECONDS,
    TRUE_MOA_OUTCOME_MAX_PLAINTEXT_BYTES,
    TRUE_MOA_OUTCOME_MAX_TEXT_BYTES,
    TRUE_MOA_OUTCOME_MAX_TTL_SECONDS,
    TRUE_MOA_OUTCOME_MAX_VERIFICATION_BYTES,
    TrueMoAOutcomeBindingError,
    TrueMoAOutcomeError,
    TrueMoAOutcomeUnavailableError,
    default_true_moa_durable_path,
    project_true_moa_completed_outcome,
    project_true_moa_outcome_binding,
)
from xiaoban.trusted_runtime.true_moa_durable_usage import project_true_moa_usage
from xiaoban.trusted_runtime.usage_drain_lease import (
    _TrueMoAUsageDrainLeaseMixin,
)


class TrueMoADurableStore(
    _TrueMoAOutcomeMixin,
    _TrueMoAAccountingMixin,
    _TrueMoAUsageDrainLeaseMixin,
    _TrueMoADurableBase,
):
    """SQLite ledger with monotonic usage and separately-keyed outcomes."""


__all__ = [
    "TRUE_MOA_COMPLETED_OUTCOME_SCHEMA",
    "TRUE_MOA_DURABLE_MAX_CALLS",
    "TRUE_MOA_DURABLE_MAX_FINAL_CALLS",
    "TRUE_MOA_DURABLE_MAX_ROWS",
    "TRUE_MOA_DURABLE_USAGE_MAX_BYTES",
    "TRUE_MOA_OUTCOME_BINDING_SCHEMA",
    "TRUE_MOA_OUTCOME_DEFAULT_TTL_SECONDS",
    "TRUE_MOA_OUTCOME_MAX_PLAINTEXT_BYTES",
    "TRUE_MOA_OUTCOME_MAX_TEXT_BYTES",
    "TRUE_MOA_OUTCOME_MAX_TTL_SECONDS",
    "TRUE_MOA_OUTCOME_MAX_VERIFICATION_BYTES",
    "TrueMoAOutcomeBindingError",
    "TrueMoAOutcomeError",
    "TrueMoAOutcomeUnavailableError",
    "TrueMoADurableStore",
    "default_true_moa_durable_path",
    "project_true_moa_completed_outcome",
    "project_true_moa_outcome_binding",
    "project_true_moa_usage",
]
