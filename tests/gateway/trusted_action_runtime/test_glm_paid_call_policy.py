from xiaoban.trusted_runtime.paid_call_policy import (
    GLM_5_3_FLASH_POLICY_REVISION,
    resolve_signed_mystand_agent_policy,
)


def test_glm_5_3_flash_policy_resolves_exact_route():
    policy = resolve_signed_mystand_agent_policy(
        GLM_5_3_FLASH_POLICY_REVISION
    )
    assert policy.provider == "zai"
    assert policy.model == "glm-5.3-flash"
