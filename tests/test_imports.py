"""Smoke test ensuring the helper module imports correctly."""

def test_import_ringaug_helper():
    # Smoke-check that helper module import succeeds.
    # Import-only test guards against packaging/path regressions.
    import ringaug.helper  # noqa: F401
