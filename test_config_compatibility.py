"""Test script to verify configuration backward compatibility.

This script tests the configuration system with both old and new environment variable formats
to ensure backward compatibility is maintained.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_settings_with_old_env_vars():
    """Test loading settings with old environment variable names (without FACECV_ prefix)."""
    for key in list(os.environ.keys()):
        if key.startswith("FACECV_") or key in ["DB_TYPE", "MYSQL_HOST", "MYSQL_USER", "MYSQL_PASSWORD"]:
            del os.environ[key]
    
    os.environ["DB_TYPE"] = "sqlite"
    os.environ["MYSQL_HOST"] = "test-host-old"
    os.environ["MYSQL_USER"] = "test-user-old"
    os.environ["MYSQL_PASSWORD"] = "test-password-old"
    
    import importlib
    if 'facecv.config.database' in sys.modules:
        importlib.reload(sys.modules['facecv.config.database'])
    
    from facecv.config.database import DatabaseConfig
    
    db_config = DatabaseConfig.from_env()
    print("\n=== Testing with OLD environment variables ===")
    print(f"DB Type: {db_config.db_type}")
    print(f"MySQL Host: {db_config.mysql_host}")
    print(f"MySQL User: {db_config.mysql_user}")
    print(f"MySQL Password: {'*' * len(db_config.mysql_password)}")
    
    assert db_config.db_type == "sqlite", f"Expected sqlite, got {db_config.db_type}"
    assert db_config.mysql_host == "test-host-old", f"Expected test-host-old, got {db_config.mysql_host}"
    assert db_config.mysql_user == "test-user-old", f"Expected test-user-old, got {db_config.mysql_user}"
    assert db_config.mysql_password == "test-password-old", f"Expected test-password-old, got {db_config.mysql_password}"
    
    print("✅ Old environment variables test passed!")
    return True

def test_settings_with_new_env_vars():
    """Test loading settings with new environment variable names (with FACECV_ prefix)."""
    os.environ["FACECV_DB_TYPE"] = "mysql"
    os.environ["FACECV_MYSQL_HOST"] = "test-host-new"
    os.environ["FACECV_MYSQL_USER"] = "test-user-new"
    os.environ["FACECV_MYSQL_PASSWORD"] = "test-password-new"
    
    for key in ["DB_TYPE", "MYSQL_HOST", "MYSQL_USER", "MYSQL_PASSWORD"]:
        if key in os.environ:
            del os.environ[key]
    
    from facecv.config.database import get_standardized_db_config
    
    db_config = get_standardized_db_config()
    print("\n=== Testing with NEW environment variables ===")
    print(f"DB Type: {db_config.db_type}")
    print(f"MySQL Host: {db_config.mysql_host}")
    print(f"MySQL User: {db_config.mysql_user}")
    print(f"MySQL Password: {'*' * len(db_config.mysql_password)}")
    
    assert db_config.db_type == "mysql", f"Expected mysql, got {db_config.db_type}"
    assert db_config.mysql_host == "test-host-new", f"Expected test-host-new, got {db_config.mysql_host}"
    assert db_config.mysql_user == "test-user-new", f"Expected test-user-new, got {db_config.mysql_user}"
    assert db_config.mysql_password == "test-password-new", f"Expected test-password-new, got {db_config.mysql_password}"
    
    print("✅ New environment variables test passed!")
    return True

def test_runtime_config():
    """Test the runtime configuration system."""
    from facecv.config.runtime_config import get_runtime_config
    
    runtime_config = get_runtime_config()
    original_value = runtime_config.get("insightface_model_pack")
    
    runtime_config.set("insightface_model_pack", "test-model-pack")
    
    assert runtime_config.get("insightface_model_pack") == "test-model-pack", \
        f"Expected test-model-pack, got {runtime_config.get('insightface_model_pack')}"
    
    print("\n=== Testing Runtime Configuration ===")
    print(f"Original value: {original_value}")
    print(f"New value: {runtime_config.get('insightface_model_pack')}")
    print("✅ Runtime configuration test passed!")
    
    runtime_config.set("insightface_model_pack", original_value)
    return True

def test_model_config_loading():
    """Test loading model configuration from YAML."""
    from facecv.config.settings import load_model_config
    
    model_config = load_model_config()
    
    print("\n=== Testing Model Configuration Loading ===")
    print(f"Model config keys: {list(model_config.keys())}")
    
    assert "insightface" in model_config, "Expected 'insightface' in model config"
    
    print("✅ Model configuration loading test passed!")
    return True

def main():
    """Run all configuration tests."""
    print("Starting configuration compatibility tests...")
    
    old_vars_test = test_settings_with_old_env_vars()
    new_vars_test = test_settings_with_new_env_vars()
    runtime_test = test_runtime_config()
    model_test = test_model_config_loading()
    
    print("\n=== Test Summary ===")
    print(f"Old environment variables test: {'✅ PASSED' if old_vars_test else '❌ FAILED'}")
    print(f"New environment variables test: {'✅ PASSED' if new_vars_test else '❌ FAILED'}")
    print(f"Runtime configuration test: {'✅ PASSED' if runtime_test else '❌ FAILED'}")
    print(f"Model configuration test: {'✅ PASSED' if model_test else '❌ FAILED'}")
    
    all_passed = all([old_vars_test, new_vars_test, runtime_test, model_test])
    print(f"\nOverall result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
