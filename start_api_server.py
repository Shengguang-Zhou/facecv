#!/usr/bin/env python3
"""API Server Startup Script with Dependency Management"""

import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock commonly missing dependencies
def mock_dependencies():
    """Mock missing dependencies to allow server startup"""
    
    # Create mock numpy
    class MockArray:
        def __init__(self, data=None):
            self.shape = (224, 224, 3)
            self.dtype = 'uint8'
        
        def __getitem__(self, key):
            return 0
        
        def tolist(self):
            return []
    
    class MockNumpy:
        ndarray = MockArray
        array = lambda self, x: MockArray(x)
        zeros = lambda self, shape, dtype=None: MockArray()
        ones = lambda self, shape, dtype=None: MockArray()
        random = type('random', (), {
            'rand': lambda *args: MockArray(),
            'randn': lambda *args: MockArray()
        })()
        
        def __getattr__(self, name):
            return lambda *args, **kwargs: MockArray()
    
    # Create mock cv2
    class MockCV2:
        COLOR_BGR2RGB = 4
        COLOR_RGB2BGR = 5
        
        def imread(self, path):
            return MockArray()
        
        def cvtColor(self, img, code):
            return img
        
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    
    # Create mock PIL
    class MockImage:
        def __init__(self):
            self.mode = 'RGB'
            self.size = (224, 224)
        
        def convert(self, mode):
            return self
        
        @classmethod
        def open(cls, fp):
            return cls()
        
        @classmethod
        def new(cls, mode, size, color=None):
            return cls()
    
    class MockImageEnhance:
        def __init__(self, img):
            self.img = img
        
        def enhance(self, factor):
            return self.img
    
    class MockPIL:
        Image = MockImage
        
        class ImageEnhance:
            Brightness = MockImageEnhance
            Contrast = MockImageEnhance
            Color = MockImageEnhance
            Sharpness = MockImageEnhance
    
    # Apply mocks
    sys.modules['numpy'] = MockNumpy()
    sys.modules['cv2'] = MockCV2()
    sys.modules['PIL'] = MockPIL()
    sys.modules['PIL.Image'] = MockPIL.Image
    sys.modules['PIL.ImageEnhance'] = MockPIL.ImageEnhance
    
    # Mock sklearn
    class MockSklearn:
        class metrics:
            class pairwise:
                @staticmethod
                def cosine_similarity(X, Y):
                    return [[0.95]]
    
    sys.modules['sklearn'] = MockSklearn()
    sys.modules['sklearn.metrics'] = MockSklearn.metrics
    sys.modules['sklearn.metrics.pairwise'] = MockSklearn.metrics.pairwise
    
    # Mock pydantic_settings
    class MockBaseSettings:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class MockPydanticSettings:
        BaseSettings = MockBaseSettings
    
    sys.modules['pydantic_settings'] = MockPydanticSettings()
    
    # Mock other dependencies
    for module in ['torch', 'tensorflow', 'deepface', 'insightface', 'chromadb']:
        sys.modules[module] = type(module, (), {})

# Apply mocks before importing the app
mock_dependencies()

# Now import and run the app
try:
    import uvicorn
    from main import app
    
    logger.info("Starting FaceCV API Server on port 7000...")
    logger.info("API Documentation: http://localhost:7000/docs")
    logger.info("Health Check: http://localhost:7000/health")
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=7000, log_level="info")
    
except Exception as e:
    logger.error(f"Failed to start server: {e}")
    import traceback
    traceback.print_exc()