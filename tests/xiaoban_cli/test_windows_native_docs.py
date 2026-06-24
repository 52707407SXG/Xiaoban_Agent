from pathlib import Path


def test_windows_native_install_path_docs_match_installer() -> None:
    doc = Path("website/docs/user-guide/windows-native.md").read_text()
    install = Path("scripts/install.ps1").read_text()

    assert "%LOCALAPPDATA%\\xiaoban\\xiaoban-agent\\venv\\Scripts" in doc
    assert "Get-Command xiaoban        # should print C:\\Users\\<you>\\AppData\\Local\\xiaoban\\xiaoban-agent\\venv\\Scripts\\xiaoban.exe" in doc
    assert '$xiaobanBin = "$InstallDir\\venv\\Scripts"' in install
