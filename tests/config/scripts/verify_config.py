"""Test script for configuration system"""

import os
from facecv.config import get_settings, get_db_config, get_runtime_config, load_model_config

def test_settings():
    """Test settings loading from environment variables"""
    settings = get_settings()
    print("\n=== Settings Test ===")
    print(f"Host: {settings.host}")
    print(f"Port: {settings.port}")
    print(f"Debug: {settings.debug}")
    return settings

def test_db_config():
    """Test database configuration"""
    db_config = get_db_config()
    print("\n=== Database Config Test ===")
    print(f"Database Type: {db_config.db_type}")
    print(f"SQLite Path: {db_config.get_sqlite_path()}")
    print(f"MySQL Connection URL: {db_config.get_connection_url()}")
    return db_config

def test_runtime_config():
    """Test runtime configuration"""
    runtime_config = get_runtime_config()
    print("\n=== Runtime Config Test ===")
    
    initial_model = runtime_config.get("insightface_model_pack")
    print(f"Initial Model Pack: {initial_model}")
    
    runtime_config.set("insightface_model_pack", "buffalo_m")
    print(f"Updated Model Pack: {runtime_config.get('insightface_model_pack')}")
    
    runtime_config.reset()
    print(f"After Reset Model Pack: {runtime_config.get('insightface_model_pack')}")
    return runtime_config

def test_model_config():
    """Test model configuration loading"""
    try:
        model_config = load_model_config()
        print("\n=== Model Config Test ===")
        print(f"Model Config Keys: {list(model_config.keys())}")
        return model_config
    except Exception as e:
        print(f"Error loading model config: {e}")
        return None

if __name__ == "__main__":
    print("Testing FaceCV Configuration System")
    
    settings = test_settings()
    
    db_config = test_db_config()
    
    runtime_config = test_runtime_config()
    
    model_config = test_model_config()
    
    print("\n=== Configuration Test Complete ===")
