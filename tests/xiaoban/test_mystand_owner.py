from xiaoban.mystand_owner import (
    configured_mystand_owner_user_id,
    is_configured_mystand_owner,
)


def test_configured_owner_identity_matches_exactly():
    env = {"MYSTAND_XIAOBAN_OWNER_USER_ID": "owner-user-001"}

    assert configured_mystand_owner_user_id(env) == "owner-user-001"
    assert is_configured_mystand_owner("owner-user-001", env) is True
    assert is_configured_mystand_owner("OWNER-USER-001", env) is False


def test_missing_or_invalid_owner_configuration_fails_closed():
    assert configured_mystand_owner_user_id({}) == ""
    assert is_configured_mystand_owner("owner-user-001", {}) is False
    assert configured_mystand_owner_user_id({
        "MYSTAND_XIAOBAN_OWNER_USER_ID": "owner user with spaces",
    }) == ""
