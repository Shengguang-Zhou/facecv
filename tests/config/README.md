# Configuration Tests

This directory contains tests for the FaceCV configuration system after the refactoring that standardized environment variables with the `FACECV_` prefix.

## Structure

- `test_configuration.py` - Unit tests for configuration classes
- `conftest.py` - Pytest fixtures for configuration tests
- `scripts/` - Verification and compatibility scripts
  - `verify_config.py` - Simple configuration verification
  - `verify_compatibility.py` - Check compatibility with old environment variables
  - `verify_env_compatibility.py` - Environment variable compatibility tests

## Running Tests

### Unit Tests
```bash
pytest tests/config/
```

### Verification Scripts
```bash
# Verify basic configuration loading
python tests/config/scripts/verify_config.py

# Check environment variable compatibility
python tests/config/scripts/verify_compatibility.py
```

## Test Results

Test results from different database backends are stored in `tests/results/`:
- `chromadb_test_results.txt`
- `mysql_test_results.txt`
- `sqlite_test_results.txt`