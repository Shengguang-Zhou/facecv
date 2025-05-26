# Configuration Management Refactor Task

## Background
Our FaceCV project currently has a terrible configuration management system with multiple issues:
1. Mixed configuration sources (hardcoded values, .env files, YAML files, Python settings)
2. Duplicate configuration definitions across multiple files
3. Inconsistent environment variable prefixes (FACECV_, no prefix, etc.)
4. Hardcoded database credentials in code
5. No clear separation between model configuration and application settings

## Objective
Refactor the entire configuration system to follow best practices:
- **Model configurations**: All model-related settings should be in `model_config.yaml`
- **Database and application settings**: All DB connections and app settings should be in `.env`
- **Single source of truth**: No duplicate configurations
- **Type safety**: Use Pydantic for validation
- **Environment awareness**: Support development/staging/production environments

## Reference Implementation
Study the EurekCV project configuration approach at `/home/a/PycharmProjects/EurekCV/config/`:
- `config.py`: Main configuration loader
- `config.yaml`: YAML-based configuration
- Environment variable usage with prefix

## Current State Analysis

### 1. Database Configuration Issues
**File**: `facecv/config/database.py`
- Hardcoded MySQL credentials (lines 25-28)
- Mixed environment variable handling
- Duplicate configuration loading logic

**File**: `facecv/config/settings.py`
- Duplicate database settings (lines 70-72)
- Duplicate ArcFace settings (lines 82-92 duplicated from 38-55)
- Mixed model configurations that should be in YAML

### 2. Model Configuration Issues
**File**: `facecv/config/model_config.yaml`
- Good structure but not consistently used across the codebase
- Contains environment-specific overrides that should be handled differently

### 3. Environment Files
**File**: `.env`
- Uses inconsistent naming (no prefix)
- Missing many required configurations

**File**: `.env.example`
- Good template but doesn't match actual usage
- Uses different variable names than code expects

## Required Changes

### 1. Create Unified Configuration System

#### A. Update `facecv/config/settings.py`:
```python
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from functools import lru_cache
import os
from pathlib import Path

class Settings(BaseSettings):
    """Application settings loaded from .env file"""
    
    # Application
    app_name: str = Field(default="FaceCV", env="FACECV_APP_NAME")
    environment: str = Field(default="production", env="FACECV_ENVIRONMENT")
    debug: bool = Field(default=False, env="FACECV_DEBUG")
    
    # API Server
    host: str = Field(default="0.0.0.0", env="FACECV_HOST")
    port: int = Field(default=7000, env="FACECV_PORT")
    workers: int = Field(default=4, env="FACECV_WORKERS")
    
    # Database
    db_type: str = Field(default="sqlite", env="FACECV_DB_TYPE")
    db_host: str = Field(default="localhost", env="FACECV_DB_HOST")
    db_port: int = Field(default=3306, env="FACECV_DB_PORT")
    db_user: str = Field(default="root", env="FACECV_DB_USER")
    db_password: str = Field(..., env="FACECV_DB_PASSWORD")  # Required
    db_name: str = Field(default="facecv", env="FACECV_DB_NAME")
    
    # Paths
    data_dir: Path = Field(default="./data", env="FACECV_DATA_DIR")
    models_dir: Path = Field(default="./models", env="FACECV_MODELS_DIR")
    upload_dir: Path = Field(default="./uploads", env="FACECV_UPLOAD_DIR")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
```

#### B. Update `facecv/config/model_config.py`:
```python
import yaml
from pathlib import Path
from typing import Dict, Any
from functools import lru_cache

@lru_cache()
def load_model_config() -> Dict[str, Any]:
    """Load model configuration from YAML file"""
    config_path = Path(__file__).parent / "model_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
```

#### C. Create `facecv/config/__init__.py`:
```python
from .settings import Settings, get_settings
from .model_config import load_model_config

__all__ = ['Settings', 'get_settings', 'load_model_config']
```

### 2. Update Environment Files

#### A. Create proper `.env` template:
```env
# Application Settings
FACECV_APP_NAME=FaceCV
FACECV_ENVIRONMENT=development
FACECV_DEBUG=true

# API Configuration
FACECV_HOST=0.0.0.0
FACECV_PORT=7000
FACECV_WORKERS=4

# Database Configuration
FACECV_DB_TYPE=mysql
FACECV_DB_HOST=eurekailab.mysql.rds.aliyuncs.com
FACECV_DB_PORT=3306
FACECV_DB_USER=root
FACECV_DB_PASSWORD=Zsg20010115_
FACECV_DB_NAME=facecv

# Paths
FACECV_DATA_DIR=./data
FACECV_MODELS_DIR=./models
FACECV_UPLOAD_DIR=./uploads
```

### 3. Remove All Hardcoded Values

Search and replace all hardcoded configurations:
- Database credentials
- Model paths
- API settings

### 4. Update Usage Across Codebase

#### Example updates needed:

**In `main.py`**:
```python
from facecv.config import get_settings

settings = get_settings()
# Use settings.host, settings.port, etc.
```

**In `database/factory.py`**:
```python
from facecv.config import get_settings

settings = get_settings()
# Use settings.db_type, settings.db_host, etc.
```

## Testing Strategy

### 1. Unit Tests
Create `tests/test_configuration.py`:
```python
import pytest
from facecv.config import get_settings, load_model_config

def test_settings_loading():
    """Test that settings load correctly from .env"""
    settings = get_settings()
    assert settings.app_name == "FaceCV"
    assert settings.db_password is not None  # Required field

def test_model_config_loading():
    """Test that model config loads from YAML"""
    config = load_model_config()
    assert "insightface" in config
    assert "deepface" in config

def test_environment_override():
    """Test that environment variables override defaults"""
    import os
    os.environ["FACECV_PORT"] = "8000"
    settings = get_settings.cache_clear()
    settings = get_settings()
    assert settings.port == 8000
```

### 2. Integration Tests
Test configuration in different environments:
- Development with SQLite
- Staging with MySQL
- Production with full settings

### 3. Configuration Validation
Add validation tests:
```python
def test_database_connection_string():
    """Test database connection string generation"""
    settings = get_settings()
    if settings.db_type == "mysql":
        conn_str = f"mysql+pymysql://{settings.db_user}:{settings.db_password}@{settings.db_host}:{settings.db_port}/{settings.db_name}"
        assert "Zsg20010115_" not in conn_str  # Should use env var
```

## Migration Steps

1. **Backup current configuration files**
2. **Create new configuration structure**
3. **Update all imports and usages**
4. **Test in development environment**
5. **Update documentation**
6. **Deploy to staging for testing**
7. **Roll out to production**

## Expected Outcomes

1. **Single source of truth**: Each configuration has one authoritative location
2. **Type safety**: All configurations validated by Pydantic
3. **Environment flexibility**: Easy to switch between dev/staging/prod
4. **Security**: No hardcoded credentials in code
5. **Maintainability**: Clear structure and documentation

## Additional Requirements

1. **Documentation**: Update README.md with configuration instructions
2. **Docker support**: Ensure Docker Compose uses proper env files
3. **CI/CD**: Update deployment scripts to use new configuration
4. **Secrets management**: Consider using environment-specific .env files

## Files to Update

Priority files that need immediate attention:
1. `facecv/config/settings.py` - Remove duplicates, use consistent env vars
2. `facecv/config/database.py` - Remove hardcoded credentials
3. `facecv/database/factory.py` - Use settings object
4. `main.py` - Use centralized settings
5. All API route files - Use settings for configuration

## Pull Request Requirements

Your PR should include:
1. Refactored configuration system
2. Updated .env.example with all required variables
3. Migration guide for existing deployments
4. Unit tests for configuration loading
5. Integration tests for different environments
6. Updated documentation

## Success Criteria

1. No hardcoded configuration values in code
2. All tests pass with new configuration system
3. Easy to deploy in different environments
4. Clear separation of concerns (app settings vs model config)
5. Improved developer experience with type hints and validation

Please study the reference code, implement these changes, and create a PR that addresses all these issues. The configuration system should be clean, maintainable, and follow best practices.