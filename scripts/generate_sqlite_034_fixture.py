"""Create tests/fixtures/sqlite_0_3_4/chatflow-0.3.4.db with chatflow-agent 0.3.4.

Pytest does not run this script. It needs network access once, to install the
pinned release into a temporary virtualenv. From the repository root:

    python scripts/generate_sqlite_034_fixture.py
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PACKAGE_SPEC = "chatflow-agent==0.3.4"
ROOT = Path(__file__).resolve().parent.parent
WRITER = Path(__file__).resolve().parent / "sqlite_034_fixture_writer.py"
DEST_DIR = ROOT / "tests" / "fixtures" / "sqlite_0_3_4"
DEST_DB = DEST_DIR / "chatflow-0.3.4.db"
DEST_README = DEST_DIR / "README.md"


def _venv_python(venv: Path) -> Path:
    """Return the venv interpreter for Unix or Windows layouts."""
    unix = venv / "bin" / "python"
    windows = venv / "Scripts" / "python.exe"
    if unix.exists():
        return unix
    if windows.exists():
        return windows
    # Prefer the platform-default path so the FileNotFoundError names the expected layout.
    return windows if sys.platform == "win32" else unix


def _isolated_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _run(cmd: list[str], env: dict[str, str], cwd: Path) -> str:
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout


def _schema_sql(db_path: Path) -> str:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT sql FROM sqlite_master
            WHERE sql IS NOT NULL
            ORDER BY type, name
            """
        ).fetchall()
    finally:
        conn.close()
    return "\n\n".join(row[0].strip() for row in rows) + "\n"


def _pragma(db_path: Path, name: str) -> str:
    conn = sqlite3.connect(db_path)
    try:
        value = conn.execute(f"PRAGMA {name}").fetchone()[0]
    finally:
        conn.close()
    return str(value)


def _render_readme(
    *,
    generated: str,
    python_version: str,
    package_version: str,
    user_version: str,
    journal_mode: str,
    integrity: str,
    schema_sql: str,
) -> str:
    return f"""# SQLite fixture from chatflow-agent 0.3.4

`chatflow-0.3.4.db` is a session database written by chatflow-agent {package_version}.
Regression tests copy it with `shutil.copy` and open the copy with the current
`SQLiteSessionStore`. Pytest does not install 0.3.4.

## Provenance

The package is `chatflow-agent=={package_version}`.
Generate the file with `python scripts/generate_sqlite_034_fixture.py`.
This copy was generated at {generated} with Python {python_version}.

`PRAGMA user_version` is {user_version}. `PRAGMA journal_mode` is {journal_mode}.
`PRAGMA integrity_check` returned `{integrity}`.

The generator installs `{PACKAGE_SPEC}` into a temporary virtualenv and runs
`scripts/sqlite_034_fixture_writer.py` with that interpreter. The writer exits
if the imported package version is not {package_version}. The writer inserts
rows with `SQLiteSessionStore.get_or_create`, `set_active_agent`, `add_message`,
and `record_processed_message`, then runs `PRAGMA wal_checkpoint(TRUNCATE)`.
The generator copies the single database file.
Timestamps stored inside `message_json` are fixed.
`created_at`, `updated_at`, and `processed_at` use SQLite `CURRENT_TIMESTAMP`
at generation time.

## Schema

These are the `sqlite_master` SQL texts in the file after chatflow-agent
{package_version} `_init_db` and the inserts. SQLite added `sqlite_sequence`
because `chatflow_messages.id` is `AUTOINCREMENT`. The compat tests compare
`sqlite_master` on a copy before and after the current store opens it.

```sql
{schema_sql.rstrip()}
```

## Rows

`legacy-billing` was created with agent Triage, then `set_active_agent`
switched it to Billing. `active_agent_name` is `Billing`. `metadata_json` is
`{{"channel": "whatsapp", "locale": "en"}}`. History has two messages.

1. Role `user`, content `Need invoice 1042`, timestamp `2026-10-23T12:00:00Z`.
2. Role `model`, content `Invoice 1042 is open.`, timestamp `2026-10-23T12:00:01Z`.

`legacy-triage` keeps agent Triage. `metadata_json` is `{{"channel": "cli"}}`.
History is one `user` message, `Hello`, timestamp `2026-10-23T12:00:02Z`.

`chatflow_processed_messages` contains `wamid.legacy.1`.

## Regenerate

From the repository root, with network access so pip can install `{PACKAGE_SPEC}`:

```
python scripts/generate_sqlite_034_fixture.py
```

The command replaces `chatflow-0.3.4.db` and rewrites this README from the new file.
"""


def main() -> None:
    if not WRITER.is_file():
        raise SystemExit(f"missing writer: {WRITER}")

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with tempfile.TemporaryDirectory(prefix="chatflow-034-") as temp_name:
        temp = Path(temp_name)
        venv = temp / "venv"
        subprocess.check_call([sys.executable, "-m", "venv", str(venv)])
        python = _venv_python(venv)
        env = _isolated_env()
        subprocess.check_call(
            [str(python), "-m", "pip", "install", "--disable-pip-version-check", PACKAGE_SPEC],
            cwd=temp,
            env=env,
        )
        produced = temp / "chatflow-0.3.4.db"
        subprocess.check_call([str(python), str(WRITER), str(produced)], cwd=temp, env=env)
        python_version = _run(
            [str(python), "-c", "import sys; print(sys.version.split()[0])"],
            env,
            temp,
        ).strip()
        package_version = _run(
            [
                str(python),
                "-c",
                "import chatflow_agent; print(chatflow_agent.__version__)",
            ],
            env,
            temp,
        ).strip()
        if package_version != "0.3.4":
            raise SystemExit(f"installed {package_version}, expected 0.3.4")
        if not produced.is_file():
            raise SystemExit(f"writer did not create {produced}")

        user_version = _pragma(produced, "user_version")
        journal_mode = _pragma(produced, "journal_mode")
        integrity = _pragma(produced, "integrity_check")
        schema_sql = _schema_sql(produced)
        readme = _render_readme(
            generated=generated,
            python_version=python_version,
            package_version=package_version,
            user_version=user_version,
            journal_mode=journal_mode,
            integrity=integrity,
            schema_sql=schema_sql,
        )

        DEST_DIR.mkdir(parents=True, exist_ok=True)
        DEST_DB.write_bytes(produced.read_bytes())
        DEST_README.write_text(readme, encoding="utf-8")

    print(f"wrote {DEST_DB}")
    print(f"wrote {DEST_README}")


if __name__ == "__main__":
    main()
