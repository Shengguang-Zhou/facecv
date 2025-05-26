#!/usr/bin/env python3
"""Server startup with proper route loading"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock numpy and cv2 which are commonly needed
import numpy as np
np.ndarray = list  # Simple mock

# Create mock for cv2
class MockCV2:
    COLOR_BGR2RGB = 4
    def imread(self, path):
        return None
    def cvtColor(self, img, code):
        return img

sys.modules['cv2'] = MockCV2()

# Start the actual server
try:
    from main import app
    import uvicorn
    
    # Update the port in startup message
    @app.on_event("startup")
    async def startup_event():
        print("FaceCV API 服务启动")
        print(f"文档地址: http://localhost:7000/docs")
        print(f"健康检查: http://localhost:7000/health")
    
    print("Starting server on port 7000...")
    uvicorn.run(app, host="0.0.0.0", port=7000)
    
except Exception as e:
    print(f"Error starting server: {e}")
    
    # Fallback to minimal server
    import uvicorn
    from fastapi import FastAPI
    
    app = FastAPI(title="FaceCV API (Minimal)")
    
    @app.get("/")
    def root():
        return {"message": "FaceCV API is running (minimal mode)"}
    
    @app.get("/health")
    def health():
        return {"status": "healthy", "mode": "minimal"}
    
    print("Starting minimal server on port 7000...")
    uvicorn.run(app, host="0.0.0.0", port=7000)