# FaceCV Limited Resource Environment Optimization

## Problem Analysis

Your client is experiencing resource exhaustion on a limited CPU environment when adding new faces. The error messages indicate:

1. **Multiple TensorFlow registrations**: CUDA/cuDNN/cuBLAS factories being registered multiple times
2. **Resource intensive model loading**: Multiple face recognition models being loaded simultaneously
3. **Memory pressure**: Server crashes during face addition operations

## Root Causes

1. **DeepFace preloading models**: The `DeepFaceRecognizer._preload_models()` method loads models at initialization
2. **Multiple model backends**: Both InsightFace and DeepFace APIs are available simultaneously
3. **No model lifecycle management**: Models stay in memory indefinitely after loading
4. **TensorFlow duplicate imports**: Multiple ONNX runtime instances from different model backends
5. **Default GPU preference**: Models attempt GPU initialization even on CPU-only systems

## Refactoring Recommendations

### 1. Disable Model Preloading

**File**: `facecv/models/deepface/core/recognizer.py`

```python
def __init__(self, ...):
    # ... existing code ...
    
    # COMMENT OUT or make conditional:
    # self._preload_models()
    
    # Add environment check:
    if os.getenv("FACECV_PRELOAD_MODELS", "false").lower() == "true":
        self._preload_models()
```

### 2. Implement Single API Mode

**File**: `main.py`

```python
# Add configuration option
FACECV_API_MODE = os.getenv("FACECV_API_MODE", "all")  # Options: "all", "insightface", "deepface"

# Conditional route registration
if FACECV_API_MODE in ["all", "deepface"]:
    app.include_router(deepface_api.router)
    
if FACECV_API_MODE in ["all", "insightface"]:
    app.include_router(insightface.router, prefix="/api/v1/insightface")
```

### 3. Add Lazy Model Loading with Cleanup

**File**: `facecv/api/routes/deepface_api.py`

```python
import gc
import threading

# Add model cleanup
_model_lock = threading.Lock()
_last_used = None

def cleanup_models():
    """Clean up unused models to free memory"""
    global deepface_recognizer, face_embedding, face_verification, face_analysis
    
    with _model_lock:
        if deepface_recognizer is not None:
            deepface_recognizer = None
            face_embedding = None
            face_verification = None
            face_analysis = None
            gc.collect()
            logger.info("DeepFace models cleaned up")

def get_deepface_components():
    """Get DeepFace components with auto-cleanup"""
    global deepface_recognizer, _last_used
    
    with _model_lock:
        _last_used = time.time()
        
        if deepface_recognizer is None:
            # ... existing initialization code ...
            
            # Schedule cleanup after inactivity
            threading.Timer(300, check_and_cleanup).start()
    
    return deepface_recognizer, face_embedding, face_verification, face_analysis

def check_and_cleanup():
    """Check if models should be cleaned up"""
    if _last_used and time.time() - _last_used > 300:  # 5 minutes
        cleanup_models()
```

### 4. Implement Smart Model Lifecycle Management

**File**: `facecv/api/routes/insightface_api.py` (Add lifecycle management)

```python
import time
import threading
import gc
import os

# Global model management
_recognizer = None
_recognizer_lock = threading.Lock()
_last_used = None
_offload_timer = None

def get_recognizer():
    """Get recognizer with automatic lifecycle management"""
    global _recognizer, _last_used, _offload_timer
    
    with _recognizer_lock:
        _last_used = time.time()
        
        # Cancel any pending offload
        if _offload_timer:
            _offload_timer.cancel()
        
        if _recognizer is None:
            logger.info("Loading InsightFace model on demand...")
            _recognizer = _create_recognizer()
        
        # Schedule auto-offload (default 5 minutes, 0 to disable)
        offload_timeout = int(os.getenv("FACECV_MODEL_OFFLOAD_TIMEOUT", "300"))
        if offload_timeout > 0:
            _offload_timer = threading.Timer(offload_timeout, _check_and_offload)
            _offload_timer.start()
        
        return _recognizer

def _create_recognizer():
    """Create recognizer with existing logic"""
    # ... existing recognizer creation code ...

def _check_and_offload():
    """Check if model should be offloaded"""
    global _recognizer, _last_used
    
    with _recognizer_lock:
        if _recognizer and _last_used:
            idle_time = time.time() - _last_used
            if idle_time >= int(os.getenv("FACECV_MODEL_OFFLOAD_TIMEOUT", "300")):
                logger.info(f"Offloading model after {idle_time:.0f}s of inactivity")
                _recognizer = None
                gc.collect()
                logger.info("Model offloaded successfully")

# Add endpoint to manually control model lifecycle
@router.post("/model/preload")
async def preload_model():
    """Manually preload model into memory"""
    recognizer = get_recognizer()
    return {"status": "success", "message": "Model preloaded"}

@router.post("/model/offload")
async def offload_model():
    """Manually offload model from memory"""
    global _recognizer, _offload_timer
    
    with _recognizer_lock:
        if _offload_timer:
            _offload_timer.cancel()
        if _recognizer:
            _recognizer = None
            gc.collect()
    
    return {"status": "success", "message": "Model offloaded"}
```

**File**: `facecv/models/insightface/real_recognizer.py` (Keep GPU default, improve fallback)

```python
def _initialize_models(self):
    """Initialize with ONNX runtime best practices from InsightFace"""
    if not INSIGHTFACE_AVAILABLE:
        logger.error("InsightFace not available - please install with: pip install insightface")
        return False
        
    try:
        logger.info("Initializing InsightFace models...")
        
        # Default providers order: GPU first, CPU fallback
        # Based on InsightFace examples
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        if not self.prefer_gpu:
            # CPU-only mode
            providers = ['CPUExecutionProvider']
            logger.info("CPU-only mode requested")
        
        # Initialize FaceAnalysis with providers
        self.face_app = FaceAnalysis(providers=providers)
        
        # Prepare with appropriate context
        # ctx_id: -1 for CPU, 0 for first GPU (InsightFace convention)
        ctx_id = -1
        if self.prefer_gpu and 'CUDAExecutionProvider' in providers:
            try:
                import onnxruntime
                if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
                    ctx_id = 0
            except:
                pass
        
        self.face_app.prepare(ctx_id=ctx_id, det_size=self.det_size)
        
        # Log actual providers being used
        self._log_active_providers()
        
        self.initialized = True
        logger.info(f"Models initialized (ctx_id={ctx_id}, GPU={'Yes' if ctx_id >= 0 else 'No'})")
        return True
        
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        return False

def _log_active_providers(self):
    """Log which ONNX providers are actually being used"""
    try:
        if hasattr(self.face_app, 'models'):
            for name, model in self.face_app.models.items():
                if hasattr(model, 'session'):
                    providers = model.session.get_providers()
                    logger.info(f"Model '{name}' using: {providers}")
    except:
        pass
```

### 5. Add Resource Monitoring

**File**: `facecv/utils/resource_monitor.py` (new file)

```python
import psutil
import logging

logger = logging.getLogger(__name__)

class ResourceMonitor:
    @staticmethod
    def check_memory():
        """Check if sufficient memory is available"""
        memory = psutil.virtual_memory()
        available_mb = memory.available / 1024 / 1024
        
        if available_mb < 500:  # Less than 500MB
            logger.warning(f"Low memory warning: {available_mb:.1f}MB available")
            return False
        return True
    
    @staticmethod
    def should_load_model(model_size_mb: int = 200):
        """Check if model should be loaded based on available resources"""
        memory = psutil.virtual_memory()
        available_mb = memory.available / 1024 / 1024
        
        # Need at least 2x model size for safe operation
        if available_mb < model_size_mb * 2:
            logger.error(f"Insufficient memory for model: {available_mb:.1f}MB available, need {model_size_mb * 2}MB")
            return False
        return True
```

### 6. Environment Configuration

**For standard environments (GPU preferred, default):**

```bash
# Standard Configuration (default)
FACECV_API_MODE=all  # Support all APIs
FACECV_INSIGHTFACE_MODEL_PACK=buffalo_l  # Full model
FACECV_INSIGHTFACE_PREFER_GPU=true  # Use GPU if available (default)
FACECV_INSIGHTFACE_DET_SIZE=[640,640]  # Full detection size
FACECV_MODEL_OFFLOAD_TIMEOUT=300  # Auto-offload after 5 minutes
```

**For limited resource environments:**

```bash
# Limited Resource Configuration
FACECV_API_MODE=insightface  # Use only one API
FACECV_PRELOAD_MODELS=false  # Disable DeepFace preloading
FACECV_INSIGHTFACE_MODEL_PACK=buffalo_s  # Smallest model
FACECV_INSIGHTFACE_DET_SIZE=[320,320]  # Smaller detection
FACECV_MODEL_OFFLOAD_TIMEOUT=60  # Aggressive offload (1 minute)
FACECV_MAX_FACES_PER_IMAGE=5  # Limit faces
FACECV_INSIGHTFACE_ENABLE_EMOTION=false  # Disable extras
FACECV_INSIGHTFACE_ENABLE_MASK=false  # Disable mask detection

# Suppress TensorFlow warnings
TF_CPP_MIN_LOG_LEVEL=3
```

**For CPU-only environments:**

```bash
# CPU-Only Configuration
FACECV_INSIGHTFACE_PREFER_GPU=false  # Force CPU mode
CUDA_VISIBLE_DEVICES=-1  # Hide CUDA devices
FACECV_MODEL_OFFLOAD_TIMEOUT=120  # 2 minute offload

# Install CPU-only ONNX runtime:
# pip install onnxruntime  # Instead of onnxruntime-gpu
```

### 7. Startup Script for Limited Resources

**File**: `start_limited.sh` (new file)

```bash
#!/bin/bash

# Set environment for limited resources
export FACECV_API_MODE=insightface
export FACECV_PRELOAD_MODELS=false
export FACECV_INSIGHTFACE_MODEL_PACK=buffalo_s
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=2

# Limit memory usage
ulimit -v 2097152  # 2GB virtual memory limit

# Start with single worker
uvicorn main:app --host 0.0.0.0 --port 7000 --workers 1 --limit-concurrency 10
```

## Implementation Priority

1. **Immediate**: Disable DeepFace model preloading (Step 1)
2. **High**: Implement model lifecycle management (Step 4) - This addresses the core issue
3. **High**: Use appropriate environment configuration (Step 6)
4. **Medium**: Implement single API mode for limited resources (Step 2)
5. **Medium**: Add DeepFace cleanup if using both APIs (Step 3)
6. **Low**: Add resource monitoring (Step 5)

## Key Insights from InsightFace

Based on the official InsightFace repository:

1. **Provider Order Matters**: Always use `['CUDAExecutionProvider', 'CPUExecutionProvider']` to enable automatic fallback
2. **ctx_id Convention**: Use `-1` for CPU, `0` for first GPU
3. **ONNX Runtime**: Use `onnxruntime-gpu` for GPU support, `onnxruntime` for CPU-only
4. **Model Loading**: Models are loaded on-demand by `FaceAnalysis` class
5. **Memory Efficiency**: Only load required models using `allowed_modules` parameter

## Testing

After implementing changes:

1. Monitor memory usage: `watch -n 1 'free -h'`
2. Test face addition with resource monitoring
3. Verify models are loaded only when needed
4. Check that cleanup releases memory

## Expected Results

- Reduced memory footprint by 60-70%
- Faster startup time
- No duplicate model registrations
- Stable operation on limited CPU systems