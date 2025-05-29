# Deprecation Notice

## Batch Processing Endpoints (Deprecated)

The following batch processing endpoints are deprecated and will be removed in a future version:

- `POST /api/v1/batch/detect` - Batch face detection
- `POST /api/v1/batch/register` - Batch face registration
- `POST /api/v1/batch/recognize` - Batch face recognition
- `POST /api/v1/batch/verify` - Batch face verification
- `POST /api/v1/batch/analyze` - Batch face analysis

## Model Management Endpoints (Deprecated)

The following model management endpoints are deprecated and will be removed in a future version:

- `POST /api/v1/models/load` - Load model
- `POST /api/v1/models/unload` - Unload model
- `GET /api/v1/models/info/{model_name}` - Get model info
- `GET /api/v1/models/performance` - Get model performance metrics
- `GET /api/v1/models/advanced/available` - Get available advanced models
- `POST /api/v1/models/advanced/recommendations` - Get model recommendations
- `POST /api/v1/models/advanced/switch` - Smart model switching

### Migration Guide for Model Management

Model management is now handled automatically by the framework:

1. **Model Loading**: Models are loaded automatically on first use
2. **Model Unloading**: Memory management is handled by the framework
3. **Model Info**: Use configuration files and documentation
4. **Performance Monitoring**: Use dedicated monitoring tools (Prometheus, Grafana, etc.)
5. **Model Selection**: Based on configuration and best practices

#### Before (Deprecated):
```python
# Manually load model
response = requests.post('/api/v1/models/load', json={'model_name': 'buffalo_l'})

# Get model info
response = requests.get('/api/v1/models/info/buffalo_l')

# Switch models
response = requests.post('/api/v1/models/advanced/switch', 
                        json={'from_model': 'buffalo_s', 'to_model': 'buffalo_l'})
```

#### After (Recommended):
```python
# Models are loaded automatically when using recognition endpoints
# Simply specify the model in your configuration or request

# Use recognition endpoint with model parameter
response = requests.post('/api/v1/insightface/recognize',
                        files={'file': image_file},
                        data={'model': 'buffalo_l'})  # Model loaded automatically if needed
```

### Migration Guide for Batch Processing

Instead of using these batch endpoints, please implement client-side batching with the individual endpoints:

#### Before (Deprecated):
```python
# Batch detect faces
response = requests.post('/api/v1/batch/detect', files=multiple_files)
```

#### After (Recommended):
```python
# Process multiple files using individual endpoint
import asyncio
import aiohttp

async def detect_single(session, file):
    async with session.post('/api/v1/insightface/detect', data={'file': file}) as resp:
        return await resp.json()

async def batch_detect(files):
    async with aiohttp.ClientSession() as session:
        tasks = [detect_single(session, file) for file in files]
        return await asyncio.gather(*tasks)

# Usage
results = asyncio.run(batch_detect(multiple_files))
```

### Benefits of Client-Side Batching:

1. **Better Error Handling**: Individual failures don't affect the entire batch
2. **Progress Tracking**: Monitor progress of each item
3. **Flexible Retry Logic**: Retry failed items independently
4. **Resource Management**: Better control over concurrent requests
5. **Simpler API**: Reduces server-side complexity

### Benefits of the New Approach:

1. **Automatic Model Management**: No need to manually load/unload models
2. **Better Resource Utilization**: Framework handles memory management intelligently
3. **Simplified API**: Fewer endpoints to maintain and understand
4. **Configuration-Based**: Models configured through settings, not API calls
5. **Performance**: Automatic caching and optimization

### Deprecation Timeline:

- **Current Version**: Deprecated endpoints are marked with warnings
- **Next Major Version**: All deprecated endpoints will be removed

Please update your applications before the next major release.