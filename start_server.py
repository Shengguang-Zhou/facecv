#!/usr/bin/env python3
"""Simple server startup script with minimal dependencies"""

import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock missing dependencies
class MockModule:
    def __getattr__(self, name):
        return MockModule()
    
    def __call__(self, *args, **kwargs):
        return MockModule()

# Mock all potentially missing modules
modules_to_mock = [
    'PIL', 'PIL.Image', 'PIL.ImageEnhance',
    'cv2',
    'sklearn', 'sklearn.metrics', 'sklearn.metrics.pairwise',
    'numpy', 'pandas',
    'torch', 'tensorflow',
    'deepface', 'insightface',
    'chromadb',
    'pydantic_settings'
]

for module in modules_to_mock:
    sys.modules[module] = MockModule()

# Now we can import and run
try:
    import uvicorn
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    
    # Create minimal app
    app = FastAPI(
        title="FaceCV API",
        description="Professional Face Recognition API Service",
        version="0.1.0"
    )
    
    # Add CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Import routes with mocked dependencies
    from facecv.api.routes import health
    app.include_router(health.router, tags=["health"])
    
    try:
        from facecv.api.routes import face
        app.include_router(face.router, prefix="/api/v1/face_recognition_insightface", tags=["face"])
    except:
        print("Warning: Could not load face routes")
    
    try:
        from facecv.api.routes import deepface
        app.include_router(deepface.router, prefix="/api/v1/face_recognition_deepface", tags=["deepface"])
    except:
        print("Warning: Could not load deepface routes")
    
    # Start server
    print("Starting FaceCV API on http://localhost:7000")
    print("API Documentation: http://localhost:7000/docs")
    print("Health Check: http://localhost:7000/health")
    
    uvicorn.run(app, host="0.0.0.0", port=7000)
    
except Exception as e:
    print(f"Failed to start server: {e}")
    import traceback
    traceback.print_exc()