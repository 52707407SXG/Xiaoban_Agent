"""Public compatibility facade for the fixed true-MoA runtime.

The implementation is split by responsibility so gateway callers retain the
stable import surface without concentrating contracts, accounting, cancellation,
and parallel execution in one module.
"""

from xiaoban.trusted_runtime.true_moa_cancel import TrueMoACancelController
from xiaoban.trusted_runtime.true_moa_contracts import (
    DEEPSEEK_FLASH_ADVISOR_SLOT,
    DEFAULT_ADJACENT_MESSAGE_COUNT,
    DEFAULT_ADJACENT_MESSAGE_MAX_CHARS,
    DEFAULT_ADVISOR_OUTPUT_MAX_CHARS,
    DEFAULT_ADVISOR_TIMEOUT_SECONDS,
    DEFAULT_CURRENT_QUESTION_MAX_CHARS,
    FINAL_EXECUTOR_SLOT,
    GPT55_ADVISOR_SLOT,
    MODE_EPOCH_HEADER,
    MOA_PRESET_ID_HEADER,
    MOA_PRESET_REVISION_HEADER,
    REASONING_MODE_HEADER,
    TRUE_MOA_ADVISOR_INPUT_MAX_BYTES,
    TRUE_MOA_ADVISOR_OUTPUT_MAX_TOKENS,
    TRUE_MOA_ADVISOR_SHUTDOWN_GRACE_SECONDS,
    TRUE_MOA_ADVISOR_SLOTS,
    TRUE_MOA_ADVISOR_USAGE_DRAIN_TIMEOUT_SECONDS,
    TRUE_MOA_ALL_SLOTS,
    TRUE_MOA_FINAL_CALL_LIMIT,
    TRUE_MOA_FINAL_INPUT_MAX_BYTES,
    TRUE_MOA_FINAL_OUTPUT_MAX_TOKENS,
    TRUE_MOA_FINAL_PAID_CALL_POLICY,
    TRUE_MOA_FINAL_SHUTDOWN_GRACE_SECONDS,
    TRUE_MOA_FINAL_SYNTHESIS_POLICY,
    TRUE_MOA_FINAL_TIMEOUT_SECONDS,
    TRUE_MOA_MODE,
    TRUE_MOA_PRESET_ID,
    TRUE_MOA_PRESET_REVISION,
    TRUE_MOA_TOTAL_CALL_LIMIT,
    TRUE_MOA_USAGE_SCHEMA,
    AdvisorMessage,
    StrictAdvisorCaller,
    StrictAdvisorResult,
    TrueMoAAdvisorBundle,
    TrueMoAContractError,
    TrueMoACostCapError,
    TrueMoAExecutionError,
    TrueMoASlot,
    TrueMoASnapshot,
    build_minimal_advisor_messages,
    enforce_true_moa_dispatch_budget,
    enforce_true_moa_final_route,
    validate_true_moa_headers,
)
from xiaoban.trusted_runtime.true_moa_execution import run_true_moa_advisors
from xiaoban.trusted_runtime.true_moa_usage import (
    TrueMoADurableNotification,
    TrueMoAUsageLedger,
)

__all__ = [
    name
    for name in globals()
    if name.startswith("TRUE_MOA_")
    or name.endswith("_HEADER")
    or name.startswith("DEFAULT_")
    or name
    in {
        "AdvisorMessage",
        "DEEPSEEK_FLASH_ADVISOR_SLOT",
        "FINAL_EXECUTOR_SLOT",
        "GPT55_ADVISOR_SLOT",
        "StrictAdvisorCaller",
        "StrictAdvisorResult",
        "TrueMoAAdvisorBundle",
        "TrueMoACancelController",
        "TrueMoAContractError",
        "TrueMoACostCapError",
        "TrueMoADurableNotification",
        "TrueMoAExecutionError",
        "TrueMoASlot",
        "TrueMoASnapshot",
        "TrueMoAUsageLedger",
        "build_minimal_advisor_messages",
        "enforce_true_moa_dispatch_budget",
        "enforce_true_moa_final_route",
        "run_true_moa_advisors",
        "validate_true_moa_headers",
    }
]
