from gateway.platforms.api_server import _sanitize_user_visible_text


def test_sanitize_user_visible_text_redacts_local_paths_and_file_urls():
    text = (
        "Do not cite file:///root/secret.md or /opt/mystand-api/mystand.sqlite; "
        "use https://example.com/source instead."
    )

    sanitized = _sanitize_user_visible_text(text)

    assert "file://" not in sanitized
    assert "/root/" not in sanitized
    assert "/opt/" not in sanitized
    assert "本地文件链接" in sanitized
    assert "本地路径" in sanitized
    assert "https://example.com/source" in sanitized
