"""Release verification script for chatflow-agent.

Validates that a GitHub release matches repository metadata and is integrated into main
before publishing to PyPI.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[import-not-found]

# PEP 440 compliant tag pattern: vX.Y.Z with optional canonical pre/dev/post releases
TAG_PATTERN = re.compile(
    r"^v(?P<version>(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
    r"(?:(?:a|b|rc|\.post|\.dev)\d+)?)$"
)


def parse_pep440_tag(tag: str) -> str:
    """Validate tag format and extract the version string without the leading 'v'."""
    if not tag:
        raise ValueError("Release tag cannot be empty.")
    match = TAG_PATTERN.match(tag)
    if not match:
        raise ValueError(
            f"Release tag '{tag}' does not match required format 'vX.Y.Z' "
            "(with optional PEP 440 pre-release suffix, e.g. 'v0.4.0', 'v0.4.0rc1')."
        )
    return match.group("version")


def get_pyproject_version(pyproject_path: Path) -> str:
    """Read [project].version from pyproject.toml."""
    if not pyproject_path.is_file():
        raise FileNotFoundError(f"pyproject.toml not found at '{pyproject_path}'.")
    content = pyproject_path.read_text(encoding="utf-8")
    data = tomllib.loads(content)
    version = data.get("project", {}).get("version")
    if not version or not isinstance(version, str):
        raise ValueError(f"Could not find valid [project].version in '{pyproject_path}'.")
    return version.strip()


def get_init_version(init_path: Path) -> str:
    """Extract __version__ from src/chatflow_agent/__init__.py using static AST analysis."""
    if not init_path.is_file():
        raise FileNotFoundError(f"__init__.py not found at '{init_path}'.")
    content = init_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(content, filename=str(init_path))
    except SyntaxError as e:
        raise ValueError(f"Syntax error in '{init_path}': {e}") from e

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__version__":
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        return node.value.value.strip()

    raise ValueError(f"Could not find static __version__ string assignment in '{init_path}'.")


def run_git_command(args: list[str], cwd: Path) -> str:
    """Execute a git command and return its stdout, or raise RuntimeError on failure."""
    try:
        res = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.strip() if e.stderr else str(e)
        raise RuntimeError(f"Git command 'git {' '.join(args)}' failed: {stderr}") from e


def verify_git_ancestry_and_checkout(
    tag: str,
    repo_root: Path,
    target_ref: str = "refs/remotes/origin/main",
) -> tuple[str, str]:
    """Verify that tag commit matches HEAD checkout and is an ancestor of target_ref.

    Returns a tuple of (resolved_tag_commit, target_commit).
    """
    # 1. Resolve commit pointed to by tag (dereferences annotated tags via ^{commit})
    tag_commit = run_git_command(["rev-parse", "--verify", f"{tag}^{{commit}}"], cwd=repo_root)

    # 2. Resolve current checkout commit (HEAD)
    head_commit = run_git_command(["rev-parse", "--verify", "HEAD"], cwd=repo_root)

    # 3. Verify checkout matches tag commit
    if tag_commit != head_commit:
        raise RuntimeError(
            f"Checkout commit ({head_commit}) does not match commit ({tag_commit}) "
            f"resolved for release tag '{tag}'."
        )

    # 4. Resolve target reference commit (e.g. origin/main)
    target_commit = run_git_command(
        ["rev-parse", "--verify", f"{target_ref}^{{commit}}"], cwd=repo_root
    )

    # 5. Check if tag commit is an ancestor of target branch
    ancestry_check = subprocess.run(
        ["git", "merge-base", "--is-ancestor", tag_commit, target_commit],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if ancestry_check.returncode == 0:
        pass
    elif ancestry_check.returncode == 1:
        raise RuntimeError(
            f"Commit {tag_commit} for tag '{tag}' is not an ancestor of {target_ref} ({target_commit}). "
            "Releases must be tagged on commits integrated into main."
        )
    else:
        stderr = ancestry_check.stderr.strip() or "Unknown git error"
        raise RuntimeError(
            f"Git error while verifying ancestry of tag '{tag}' against {target_ref} "
            f"(exit code {ancestry_check.returncode}): {stderr}"
        )

    return tag_commit, target_commit


def verify_release(
    tag: str,
    repo_root: Path | None = None,
    target_ref: str = "refs/remotes/origin/main",
) -> None:
    """Execute all release verification checks."""
    if repo_root is None:
        repo_root = Path.cwd()

    print(f"--> Verifying release for tag: '{tag}'")
    tag_version = parse_pep440_tag(tag)
    print(f"  [OK] Tag format valid. Extracted version: '{tag_version}'")

    pyproject_path = repo_root / "pyproject.toml"
    pyproject_version = get_pyproject_version(pyproject_path)
    print(f"  [OK] pyproject.toml version: '{pyproject_version}'")

    init_path = repo_root / "src" / "chatflow_agent" / "__init__.py"
    init_version = get_init_version(init_path)
    print(f"  [OK] __init__.__version__: '{init_version}'")

    if tag_version != pyproject_version:
        raise ValueError(
            f"Tag version '{tag_version}' does not match pyproject.toml version '{pyproject_version}'."
        )

    if tag_version != init_version:
        raise ValueError(
            f"Tag version '{tag_version}' does not match chatflow_agent.__version__ '{init_version}'."
        )

    print("  [OK] Version consistency verified across tag, pyproject.toml, and __init__.py")

    tag_commit, target_commit = verify_git_ancestry_and_checkout(tag, repo_root, target_ref)
    print(f"  [OK] Checkout HEAD matches tag commit: {tag_commit[:8]}")
    print(f"  [OK] Tag commit is an ancestor of {target_ref}: {target_commit[:8]}")
    print("--> All release consistency and security checks PASSED.")


def main() -> int:
    tag = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.environ.get("RELEASE_TAG") or os.environ.get("GITHUB_REF_NAME", "")
    )
    target_ref = os.environ.get("TARGET_REF", "refs/remotes/origin/main")

    if not tag:
        sys.stderr.write(
            "ERROR: No release tag provided. Pass the tag as an argument or set RELEASE_TAG.\n"
        )
        return 1

    try:
        verify_release(tag=tag, target_ref=target_ref)
        return 0
    except Exception as exc:
        sys.stderr.write(f"\nRELEASE VERIFICATION FAILED: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
