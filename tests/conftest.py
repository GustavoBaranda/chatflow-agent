"""Shared test setup."""

import os
import tempfile

# examples/webhook_api_service.py opens its SQLite store as soon as it is imported, which happens
# when pytest collects tests/test_webhook_api_service.py. Point it at a throwaway directory before
# collection, so a test run never leaves api_sessions.db (or its -wal/-shm files) in the
# repository root (#80). The directory is removed when the test process exits.
_db_dir = tempfile.TemporaryDirectory(prefix="chatflow-tests-", ignore_cleanup_errors=True)
os.environ.setdefault("CHATFLOW_DB_PATH", os.path.join(_db_dir.name, "api_sessions.db"))
