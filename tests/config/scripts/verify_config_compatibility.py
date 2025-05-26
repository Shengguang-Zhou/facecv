"""Simple script to verify configuration backward compatibility.

This script tests the configuration system with both old and new environment variable formats
to ensure backward compatibility is maintained.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Test configuration backward compatibility."""
    print("Testing configuration backward compatibility...")
    
    print("\n=== Testing with OLD environment variables ===")
    for key in list(os.environ.keys()):
        if key.startswith("FACECV_"):
            del os.environ[key]
    
    os.environ["DB_TYPE"] = "sqlite"
    os.environ["MYSQL_HOST"] = "test-host-old"
    os.environ["MYSQL_USER"] = "test-user-old"
    os.environ["MYSQL_PASSWORD"] = "test-password-old"
    
    from facecv.config.database import DatabaseConfig
    
    db_config = DatabaseConfig.from_env()
    print(f"DB Type: {db_config.db_type}")
    print(f"MySQL Host: {db_config.mysql_host}")
    print(f"MySQL User: {db_config.mysql_user}")
    print(f"MySQL Password: {'*' * len(db_config.mysql_password)}")
    
    print("\n=== Testing with NEW environment variables ===")
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
    
    print("\nTest completed.")

if __name__ == "__main__":
    main()
