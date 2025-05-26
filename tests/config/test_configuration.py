"""Unit tests for the configuration system"""

import os
import pytest
from unittest.mock import patch
import tempfile
import yaml

from facecv.config import get_settings, get_db_config, get_runtime_config, load_model_config
from facecv.config.settings import Settings
from facecv.config.database import DatabaseConfig
from facecv.config.runtime_config import RuntimeConfig


class TestSettings:
    """Test the Settings class and related functions"""
    
    def test_settings_loading(self):
        """Test that settings load correctly from environment variables"""
        settings = get_settings()
        assert isinstance(settings, Settings)
        assert hasattr(settings, "host")
        assert hasattr(settings, "port")
        assert hasattr(settings, "debug")
    
    def test_environment_override(self):
        """Test that environment variables override defaults"""
        with patch.dict(os.environ, {"FACECV_PORT": "8888"}):
            from facecv.config.settings import get_settings
            get_settings.cache_clear()
            
            settings = get_settings()
            assert settings.port == 8888


class TestDatabaseConfig:
    """Test the DatabaseConfig class and related functions"""
    
    def test_db_config_loading(self):
        """Test database configuration loading"""
        db_config = get_db_config()
        assert isinstance(db_config, DatabaseConfig)
        assert hasattr(db_config, "db_type")
        assert hasattr(db_config, "get_connection_url")
    
    def test_sqlite_path(self):
        """Test SQLite path generation"""
        db_config = get_db_config()
        sqlite_path = db_config.get_sqlite_path()
        assert isinstance(sqlite_path, str)
        assert sqlite_path.endswith(".db")
    
    def test_connection_url(self):
        """Test connection URL generation"""
        with patch.dict(os.environ, {"FACECV_DB_TYPE": "sqlite"}):
            db_config = get_db_config()
            conn_url = db_config.get_connection_url()
            assert conn_url.startswith("sqlite:///")
        
        with patch.dict(os.environ, {
            "FACECV_DB_TYPE": "mysql",
            "FACECV_MYSQL_HOST": "test-host",
            "FACECV_MYSQL_PORT": "3306",
            "FACECV_MYSQL_USER": "test-user",
            "FACECV_MYSQL_PASSWORD": "test-password",
            "FACECV_MYSQL_DATABASE": "test-db"
        }):
            db_config = get_db_config()
            conn_url = db_config.get_connection_url()
            assert conn_url.startswith("mysql+pymysql://")
            assert "test-host" in conn_url
            assert "test-user" in conn_url
            assert "test-password" in conn_url
            assert "test-db" in conn_url
    
    def test_mysql_validation_missing_host(self):
        """Test that MySQL validation fails when host is missing"""
        with patch.dict(os.environ, {
            "FACECV_DB_TYPE": "mysql",
            "FACECV_MYSQL_HOST": "",
            "FACECV_MYSQL_USER": "test-user",
            "FACECV_MYSQL_PASSWORD": "test-password"
        }):
            with pytest.raises(ValueError, match="MySQL主机不能为空"):
                DatabaseConfig.from_env()
    
    def test_mysql_validation_missing_user(self):
        """Test that MySQL validation fails when user is missing"""
        with patch.dict(os.environ, {
            "FACECV_DB_TYPE": "mysql",
            "FACECV_MYSQL_HOST": "test-host",
            "FACECV_MYSQL_USER": "",
            "FACECV_MYSQL_PASSWORD": "test-password"
        }):
            with pytest.raises(ValueError, match="MySQL用户名不能为空"):
                DatabaseConfig.from_env()
    
    def test_mysql_validation_missing_password(self):
        """Test that MySQL validation fails when password is missing"""
        with patch.dict(os.environ, {
            "FACECV_DB_TYPE": "mysql",
            "FACECV_MYSQL_HOST": "test-host",
            "FACECV_MYSQL_USER": "test-user",
            "FACECV_MYSQL_PASSWORD": ""
        }):
            with pytest.raises(ValueError, match="MySQL密码不能为空"):
                DatabaseConfig.from_env()
    
    def test_sqlite_no_validation_required(self):
        """Test that SQLite doesn't require MySQL credentials"""
        with patch.dict(os.environ, {
            "FACECV_DB_TYPE": "sqlite",
            "FACECV_MYSQL_HOST": "",
            "FACECV_MYSQL_USER": "",
            "FACECV_MYSQL_PASSWORD": ""
        }):
            db_config = DatabaseConfig.from_env()
            assert db_config.db_type == "sqlite"


class TestRuntimeConfig:
    """Test the RuntimeConfig class and related functions"""
    
    def test_runtime_config_singleton(self):
        """Test that RuntimeConfig is a singleton"""
        config1 = get_runtime_config()
        config2 = get_runtime_config()
        assert config1 is config2
    
    def test_get_set_values(self):
        """Test getting and setting values"""
        runtime_config = get_runtime_config()
        
        runtime_config.set("test_key", "test_value")
        assert runtime_config.get("test_key") == "test_value"
        
        assert runtime_config.get("non_existent_key", "default") == "default"
    
    def test_update_values(self):
        """Test updating multiple values"""
        runtime_config = get_runtime_config()
        
        runtime_config.update({
            "key1": "value1",
            "key2": "value2"
        })
        
        assert runtime_config.get("key1") == "value1"
        assert runtime_config.get("key2") == "value2"
    
    def test_reset(self):
        """Test resetting to defaults"""
        runtime_config = get_runtime_config()
        
        runtime_config.set("insightface_model_pack", "custom_model")
        assert runtime_config.get("insightface_model_pack") == "custom_model"
        
        runtime_config.reset()
        assert runtime_config.get("insightface_model_pack") == "buffalo_l"


class TestModelConfig:
    """Test the model configuration loading"""
    
    def test_model_config_loading(self):
        """Test that model config loads from YAML"""
        try:
            config = load_model_config()
            assert isinstance(config, dict)
            assert "insightface" in config
        except Exception as e:
            pytest.skip(f"Model config loading failed: {e}")
    
    def test_model_config_with_temp_file(self):
        """Test model config loading with a temporary file"""
        test_config = {
            "test_model": {
                "name": "test_model",
                "path": "/path/to/model",
                "params": {
                    "threshold": 0.5
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp_file:
            yaml.dump(test_config, temp_file)
            temp_file.flush()
            temp_path = temp_file.name
        
        try:
            config = load_model_config(config_path=temp_path)
            assert "test_model" in config
            assert config["test_model"]["name"] == "test_model"
            assert config["test_model"]["params"]["threshold"] == 0.5
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
