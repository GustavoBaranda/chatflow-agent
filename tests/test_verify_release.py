"""Tests for scripts/verify_release.py release validation system."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.verify_release import (
    get_init_version,
    get_pyproject_version,
    main,
    parse_pep440_tag,
    verify_git_ancestry_and_checkout,
    verify_release,
)

# ---------------------------------------------------------------------------
# Unit tests: Tag parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tag", "expected_version"),
    [
        ("v0.1.0", "0.1.0"),
        ("v0.3.3", "0.3.3"),
        ("v0.4.0", "0.4.0"),
        ("v1.0.0", "1.0.0"),
        ("v0.4.0rc1", "0.4.0rc1"),
        ("v0.4.0b2", "0.4.0b2"),
        ("v0.4.0a1", "0.4.0a1"),
        ("v0.4.0.dev1", "0.4.0.dev1"),
        ("v0.4.0.post1", "0.4.0.post1"),
        ("v10.20.30rc99", "10.20.30rc99"),
    ],
)
def test_parse_pep440_tag_valid(tag: str, expected_version: str) -> None:
    assert parse_pep440_tag(tag) == expected_version


@pytest.mark.parametrize(
    "invalid_tag",
    [
        "0.4.0",  # missing leading v
        "v0.4",  # missing patch
        "v0.4.0.0",  # too many digits
        "v0.4.0-rc1",  # non-canonical PEP 440 separator
        "release-0.4.0",
        "latest",
        "vX.Y.Z",
        "v0.4.0-beta",
    ],
)
def test_parse_pep440_tag_invalid(invalid_tag: str) -> None:
    with pytest.raises(ValueError, match="does not match required format"):
        parse_pep440_tag(invalid_tag)


def test_parse_pep440_tag_empty() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        parse_pep440_tag("")


# ---------------------------------------------------------------------------
# Unit tests: Metadata extraction
# ---------------------------------------------------------------------------


def test_get_pyproject_version(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "test"\nversion = "0.4.0"\n', encoding="utf-8")
    assert get_pyproject_version(pyproject) == "0.4.0"


def test_get_pyproject_version_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        get_pyproject_version(tmp_path / "nonexistent.toml")


def test_get_pyproject_version_missing_field(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "test"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Could not find valid"):
        get_pyproject_version(pyproject)


def test_get_init_version(tmp_path: Path) -> None:
    init_file = tmp_path / "__init__.py"
    init_file.write_text('__version__ = "0.4.0"\n__all__ = ["__version__"]\n', encoding="utf-8")
    assert get_init_version(init_file) == "0.4.0"


def test_get_init_version_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        get_init_version(tmp_path / "nonexistent.py")


def test_get_init_version_missing_assignment(tmp_path: Path) -> None:
    init_file = tmp_path / "__init__.py"
    init_file.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Could not find static __version__"):
        get_init_version(init_file)


# ---------------------------------------------------------------------------
# Real Git Repository Tests
# ---------------------------------------------------------------------------

requires_git = pytest.mark.skipif(
    shutil.which("git") is None,
    reason="git executable not found in PATH",
)


def _setup_git_repo(tmp_path: Path, version: str = "0.4.0") -> Path:
    """Helper to initialize a clean git repo with project files."""
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)

    git("init", "-b", "main")
    git("config", "user.name", "Test Committer")
    git("config", "user.email", "test@example.com")

    # Write pyproject.toml and __init__.py
    pyproject = repo / "pyproject.toml"
    pyproject.write_text(f'[project]\nname = "chatflow-agent"\nversion = "{version}"\n')

    init_dir = repo / "src" / "chatflow_agent"
    init_dir.mkdir(parents=True, exist_ok=True)
    init_file = init_dir / "__init__.py"
    init_file.write_text(f'__version__ = "{version}"\n')

    git("add", ".")
    git("commit", "-m", "Initial commit on main")
    return repo


@requires_git
def test_verify_release_valid_on_main(tmp_path: Path) -> None:
    """Tag on main with matching metadata must pass."""
    repo = _setup_git_repo(tmp_path, version="0.4.0")

    # Tag the commit and checkout the tag
    subprocess.run(["git", "tag", "v0.4.0"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "checkout", "--detach", "v0.4.0"], cwd=repo, check=True, capture_output=True
    )

    # Validate against main branch
    verify_release(tag="v0.4.0", repo_root=repo, target_ref="main")


@requires_git
def test_verify_release_annotated_tag(tmp_path: Path) -> None:
    """Annotated tag (tag object) on main must be correctly dereferenced and pass."""
    repo = _setup_git_repo(tmp_path, version="0.4.0")

    subprocess.run(
        ["git", "tag", "-a", "v0.4.0", "-m", "Release v0.4.0"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", "--detach", "v0.4.0"], cwd=repo, check=True, capture_output=True
    )

    verify_release(tag="v0.4.0", repo_root=repo, target_ref="main")


@requires_git
def test_verify_release_rejects_isolated_branch_tag(tmp_path: Path) -> None:
    """Tag placed on an isolated/unmerged branch must be rejected."""
    repo = _setup_git_repo(tmp_path, version="0.4.0")

    # Create an isolated branch with an unmerged commit
    subprocess.run(
        ["git", "checkout", "-b", "isolated-branch"], cwd=repo, check=True, capture_output=True
    )
    isolated_file = repo / "isolated.txt"
    isolated_file.write_text("isolated")
    subprocess.run(["git", "add", "isolated.txt"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Commit only on isolated branch"],
        cwd=repo,
        check=True,
        capture_output=True,
    )

    # Tag on isolated branch
    subprocess.run(
        ["git", "tag", "-a", "v0.4.0", "-m", "Release on isolated"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", "--detach", "v0.4.0"], cwd=repo, check=True, capture_output=True
    )

    # In main, add another commit so branches have diverged
    subprocess.run(["git", "checkout", "main"], cwd=repo, check=True, capture_output=True)
    (repo / "main_update.txt").write_text("update")
    subprocess.run(["git", "add", "main_update.txt"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Main moves forward"], cwd=repo, check=True, capture_output=True
    )

    # Checkout back to tag commit
    subprocess.run(
        ["git", "checkout", "--detach", "v0.4.0"], cwd=repo, check=True, capture_output=True
    )

    with pytest.raises(RuntimeError, match="is not an ancestor of main"):
        verify_release(tag="v0.4.0", repo_root=repo, target_ref="main")


@requires_git
def test_verify_release_rejects_version_mismatch(tmp_path: Path) -> None:
    """Tag version mismatching pyproject.toml or __init__.py must be rejected."""
    repo = _setup_git_repo(tmp_path, version="0.3.3")

    subprocess.run(["git", "tag", "v0.4.0"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "checkout", "--detach", "v0.4.0"], cwd=repo, check=True, capture_output=True
    )

    with pytest.raises(ValueError, match="does not match pyproject.toml version"):
        verify_release(tag="v0.4.0", repo_root=repo, target_ref="main")


@requires_git
def test_verify_release_rejects_init_version_mismatch(tmp_path: Path) -> None:
    """Tag matching pyproject.toml but mismatching __init__.py must be rejected."""
    repo = _setup_git_repo(tmp_path, version="0.4.0")

    # Desynchronize __init__.py
    (repo / "src" / "chatflow_agent" / "__init__.py").write_text('__version__ = "0.3.3"\n')
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Desync __init__"], cwd=repo, check=True, capture_output=True
    )
    subprocess.run(["git", "tag", "v0.4.0"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "checkout", "--detach", "v0.4.0"], cwd=repo, check=True, capture_output=True
    )

    with pytest.raises(ValueError, match="does not match chatflow_agent.__version__"):
        verify_release(tag="v0.4.0", repo_root=repo, target_ref="main")


@requires_git
def test_verify_release_rejects_missing_target_ref(tmp_path: Path) -> None:
    """Missing target ref (e.g. un-fetched origin/main) must fail securely."""
    repo = _setup_git_repo(tmp_path, version="0.4.0")
    subprocess.run(["git", "tag", "v0.4.0"], cwd=repo, check=True, capture_output=True)

    with pytest.raises(RuntimeError, match="Git command .* failed"):
        verify_git_ancestry_and_checkout(
            tag="v0.4.0", repo_root=repo, target_ref="refs/remotes/origin/nonexistent"
        )


@requires_git
def test_verify_release_rejects_checkout_mismatch(tmp_path: Path) -> None:
    """If checkout HEAD does not match the tag commit, must fail."""
    repo = _setup_git_repo(tmp_path, version="0.4.0")

    subprocess.run(["git", "tag", "v0.4.0"], cwd=repo, check=True, capture_output=True)

    # Add a commit and stay on it, while validating tag v0.4.0
    (repo / "extra.txt").write_text("extra")
    subprocess.run(["git", "add", "extra.txt"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Commit after tag"], cwd=repo, check=True, capture_output=True
    )

    with pytest.raises(RuntimeError, match="Checkout commit .* does not match commit"):
        verify_git_ancestry_and_checkout(tag="v0.4.0", repo_root=repo, target_ref="main")


@requires_git
def test_verify_release_with_git_remote_workflow_simulation(tmp_path: Path) -> None:
    """Test full workflow simulation using a real bare Git remote and refs/remotes/origin/main."""
    remote_repo = tmp_path / "remote.git"
    remote_repo.mkdir()
    subprocess.run(["git", "init", "--bare", "-b", "main"], cwd=remote_repo, check=True, capture_output=True)

    clone_repo = tmp_path / "clone"
    subprocess.run(["git", "clone", str(remote_repo), str(clone_repo)], check=True, capture_output=True)

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=clone_repo, check=True, capture_output=True, text=True)

    git("config", "user.name", "Test Committer")
    git("config", "user.email", "test@example.com")

    # Commit initial v0.4.0 project files to main
    pyproject = clone_repo / "pyproject.toml"
    pyproject.write_text('[project]\nname = "chatflow-agent"\nversion = "0.4.0"\n', encoding="utf-8")
    init_dir = clone_repo / "src" / "chatflow_agent"
    init_dir.mkdir(parents=True, exist_ok=True)
    init_file = init_dir / "__init__.py"
    init_file.write_text('__version__ = "0.4.0"\n', encoding="utf-8")

    git("add", ".")
    git("commit", "-m", "Release 0.4.0 on main")
    git("push", "origin", "main")

    # 1. Test valid integrated release tag
    git("tag", "-a", "v0.4.0", "-m", "Release v0.4.0")
    git("push", "origin", "v0.4.0")
    git("checkout", "--detach", "v0.4.0")

    # Simulates the workflow fetch: git fetch origin main:refs/remotes/origin/main
    git("fetch", "origin", "main:refs/remotes/origin/main")
    verify_release(tag="v0.4.0", repo_root=clone_repo, target_ref="refs/remotes/origin/main")

    # 2. Test isolated branch release tag (simulating release/0.3.3 incident)
    git("checkout", "-b", "release/0.4.1-isolated")
    pyproject.write_text('[project]\nname = "chatflow-agent"\nversion = "0.4.1"\n', encoding="utf-8")
    init_file.write_text('__version__ = "0.4.1"\n', encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "Isolated release commit")
    git("tag", "-a", "v0.4.1", "-m", "Release v0.4.1 on isolated branch")

    # Advance main on remote so branches diverge
    git("checkout", "main")
    (clone_repo / "new_feature.txt").write_text("feature on main")
    git("add", "new_feature.txt")
    git("commit", "-m", "Advance main independently")
    git("push", "origin", "main")

    # Checkout the isolated tag and refresh origin/main
    git("checkout", "--detach", "v0.4.1")
    git("fetch", "origin", "main:refs/remotes/origin/main")

    with pytest.raises(RuntimeError, match="is not an ancestor of refs/remotes/origin/main"):
        verify_release(tag="v0.4.1", repo_root=clone_repo, target_ref="refs/remotes/origin/main")


@requires_git
def test_verify_git_ancestry_unexpected_git_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unexpected git exit code (> 1) must raise a distinct Git error message."""
    repo = _setup_git_repo(tmp_path, version="0.4.0")
    subprocess.run(["git", "tag", "v0.4.0"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "checkout", "--detach", "v0.4.0"], cwd=repo, check=True, capture_output=True)

    original_run = subprocess.run

    def mock_run(args, **kwargs):  # type: ignore[no-untyped-def]
        if len(args) > 1 and args[1] == "merge-base":
            return subprocess.CompletedProcess(args=args, returncode=128, stdout="", stderr="fatal: mock git crash")
        return original_run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", mock_run)

    with pytest.raises(RuntimeError, match=r"Git error while verifying ancestry of tag 'v0.4.0' against main \(exit code 128\): fatal: mock git crash"):
        verify_git_ancestry_and_checkout(tag="v0.4.0", repo_root=repo, target_ref="main")


@requires_git
def test_cli_main_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test CLI main() returns 0 on valid release."""
    repo = _setup_git_repo(tmp_path, version="0.4.0")
    subprocess.run(["git", "tag", "v0.4.0"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "checkout", "--detach", "v0.4.0"], cwd=repo, check=True, capture_output=True
    )

    monkeypatch.chdir(repo)
    monkeypatch.setenv("RELEASE_TAG", "v0.4.0")
    monkeypatch.setenv("TARGET_REF", "main")
    monkeypatch.setattr(sys, "argv", ["verify_release.py"])

    assert main() == 0


@requires_git
def test_cli_main_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test CLI main() returns 1 on error and logs to stderr."""
    repo = _setup_git_repo(tmp_path, version="0.4.0")
    monkeypatch.chdir(repo)
    monkeypatch.setenv("RELEASE_TAG", "invalid-tag")
    monkeypatch.setattr(sys, "argv", ["verify_release.py"])

    assert main() == 1


def test_cli_main_no_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test CLI main() returns 1 when no tag is supplied."""
    monkeypatch.delenv("RELEASE_TAG", raising=False)
    monkeypatch.delenv("GITHUB_REF_NAME", raising=False)
    monkeypatch.setattr(sys, "argv", ["verify_release.py"])

    assert main() == 1
