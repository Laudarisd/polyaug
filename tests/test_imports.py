"""Smoke test ensuring the helper module imports correctly."""

def test_import_polyaug_helper():
    # Smoke-check that helper module import succeeds.
    # Import-only test guards against packaging/path regressions.
    import polyaug.helper  # noqa: F401
