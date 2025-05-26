"""Test script to verify environment variable backward compatibility.

This script tests the configuration system with both old and new environment variable formats
by temporarily disabling the .env file to ensure environment variables take precedence.
"""

import os
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def backup_env_file():
    """Backup the .env file to prevent it from interfering with tests."""
    env_path = Path(__file__).parent / ".env"
    backup_path = Path(__file__).parent / ".env.backup"
    
    if env_path.exists():
        print(f"Backing up .env file to {backup_path}")
        shutil.copy(env_path, backup_path)
        env_path.rename(Path(__file__).parent / ".env.disabled")
        return True
    return False

def restore_env_file():
    """Restore the .env file after tests."""
    env_path = Path(__file__).parent / ".env"
    backup_path = Path(__file__).parent / ".env.backup"
    disabled_path = Path(__file__).parent / ".env.disabled"
    
    if disabled_path.exists():
        print(f"Restoring .env file")
        disabled_path.rename(env_path)
    
    if backup_path.exists():
        backup_path.unlink()

def test_old_env_vars():
    """Test with old environment variable names."""
    for key in list(os.environ.keys()):
        if key.startswith("FACECV_"):
            del os.environ[key]
    
    os.environ["DB_TYPE"] = "sqlite"
    os.environ["MYSQL_HOST"] = "test-host-old"
    os.environ["MYSQL_USER"] = "test-user-old"
    os.environ["MYSQL_PASSWORD"] = "test-password-old"
    
    from importlib import reload
    import facecv.config.database
    reload(facecv.config.database)
    
    from facecv.config.database import DatabaseConfig
    
    db_config = DatabaseConfig.from_env()
    print(f"DB Type: {db_config.db_type}")
    print(f"MySQL Host: {db_config.mysql_host}")
    print(f"MySQL User: {db_config.mysql_user}")
    print(f"MySQL Password: {'*' * len(db_config.mysql_password)}")
    
    success = True
    if db_config.db_type != "sqlite":
        print(f"❌ ERROR: Expected db_type=sqlite, got {db_config.db_type}")
        success = False
    if db_config.mysql_host != "test-host-old":
        print(f"❌ ERROR: Expected mysql_host=test-host-old, got {db_config.mysql_host}")
        success = False
    if db_config.mysql_user != "test-user-old":
        print(f"❌ ERROR: Expected mysql_user=test-user-old, got {db_config.mysql_user}")
        success = False
    if db_config.mysql_password != "test-password-old":
        print(f"❌ ERROR: Expected mysql_password=test-password-old, got {db_config.mysql_password}")
        success = False
    
    if success:
        print("✅ Old environment variables test passed!")
    return success

def test_new_env_vars():
    """Test with new environment variable names."""
    for key in ["DB_TYPE", "MYSQL_HOST", "MYSQL_USER", "MYSQL_PASSWORD"]:
        if key in os.environ:
            del os.environ[key]
    
    os.environ["FACECV_DB_TYPE"] = "mysql"
    os.environ["FACECV_MYSQL_HOST"] = "test-host-new"
    os.environ["FACECV_MYSQL_USER"] = "test-user-new"
    os.environ["FACECV_MYSQL_PASSWORD"] = "test-password-new"
    
    from importlib import reload
    import facecv.config.database
    reload(facecv.config.database)
    
    from facecv.config.database import DatabaseConfig
    
    db_config = DatabaseConfig.from_env()
    print(f"DB Type: {db_config.db_type}")
    print(f"MySQL Host: {db_config.mysql_host}")
    print(f"MySQL User: {db_config.mysql_user}")
    print(f"MySQL Password: {'*' * len(db_config.mysql_password)}")
    
    success = True
    if db_config.db_type != "mysql":
        print(f"❌ ERROR: Expected db_type=mysql, got {db_config.db_type}")
        success = False
    if db_config.mysql_host != "test-host-new":
        print(f"❌ ERROR: Expected mysql_host=test-host-new, got {db_config.mysql_host}")
        success = False
    if db_config.mysql_user != "test-user-new":
        print(f"❌ ERROR: Expected mysql_user=test-user-new, got {db_config.mysql_user}")
        success = False
    if db_config.mysql_password != "test-password-new":
        print(f"❌ ERROR: Expected mysql_password=test-password-new, got {db_config.mysql_password}")
        success = False
    
    if success:
        print("✅ New environment variables test passed!")
    return success

def main():
    """Test configuration backward compatibility."""
    print("Testing configuration backward compatibility...")
    
    env_backed_up = backup_env_file()
    
    try:
        print("\n=== Testing with OLD environment variables ===")
        old_vars_success = test_old_env_vars()
        
        print("\n=== Testing with NEW environment variables ===")
        new_vars_success = test_new_env_vars()
        
        print("\n=== Test Summary ===")
        print(f"Old environment variables test: {'✅ PASSED' if old_vars_success else '❌ FAILED'}")
        print(f"New environment variables test: {'✅ PASSED' if new_vars_success else '❌ FAILED'}")
        
        all_passed = old_vars_success and new_vars_success
        print(f"\nOverall result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
        
        return 0 if all_passed else 1
    finally:
        if env_backed_up:
            restore_env_file()
        print("\nTest completed.")

if __name__ == "__main__":
    sys.exit(main())
