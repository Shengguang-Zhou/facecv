"""
Pytest configuration for configuration tests
"""
import pytest
import os
from pathlib import Path
import tempfile
import yaml

@pytest.fixture(scope="session", autouse=True)
def verify_api_server():
    """Override the global fixture to allow config tests to run without API server"""
    pass

@pytest.fixture(scope="function")
def temp_env_vars():
    """Fixture to provide temporary environment variables"""
    original_env = os.environ.copy()
    yield os.environ
    os.environ.clear()
    os.environ.update(original_env)

@pytest.fixture(scope="function")
def temp_model_config():
    """Fixture to provide a temporary model config file"""
    test_config = {
        "insightface": {
            "model_pack": "buffalo_l",
            "det_size": [640, 640],
            "det_thresh": 0.5,
            "similarity_thresh": 0.35
        },
        "deepface": {
            "model": "VGG-Face",
            "detector": "opencv",
            "distance_metric": "cosine"
        }
    }
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp_file:
        yaml.dump(test_config, temp_file)
        temp_file.flush()
        yield temp_file.name
        
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)
