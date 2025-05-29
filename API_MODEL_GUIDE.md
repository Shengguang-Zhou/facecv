# InsightFace API Model Guide

## Model Architecture

InsightFace uses different models for different tasks:

### Detection-Only Models (SCRFD)

SCRFD (Sample and Computation Redistribution for Efficient Face Detection) models are specialized for face detection only:

| Model | Description | FLOPs | Use Case |
|-------|-------------|-------|----------|
| `scrfd_10g` | SCRFD-10GF | 10G | Highest accuracy detection, production |
| `scrfd_2.5g` | SCRFD-2.5GF | 2.5G | Balanced accuracy/speed |
| `scrfd_500m` | SCRFD-500MF | 500M | Fast detection, mobile |
| `scrfd_10g_bnkps` | SCRFD-10G with keypoints | 10G | High accuracy with facial keypoints |
| `scrfd_2.5g_bnkps` | SCRFD-2.5G with keypoints | 2.5G | Balanced with facial keypoints |

### Full Model Packages

Complete packages that include detection + recognition + analysis:

| Model | Detection | Recognition | Description |
|-------|-----------|-------------|-------------|
| `buffalo_l` | SCRFD-10GF | ResNet50 | Best accuracy, production ready |
| `buffalo_m` | SCRFD-2.5GF | ResNet50 | Balanced performance |
| `buffalo_s` | SCRFD-500MF | MobileFaceNet | Fast, mobile optimized |
| `antelopev2` | SCRFD-10G-BNKPS | GlinTR100 | Research grade, highest accuracy |

## API Endpoint Model Usage

### `/api/v1/insightface/detect`
- **Recommended models**: SCRFD variants (`scrfd_10g`, `scrfd_2.5g`, `scrfd_500m`)
- **Why**: Detection-only task, no need for recognition models
- **Example**:
  ```bash
  curl -X POST /api/v1/insightface/detect \
    -F "file=@image.jpg" \
    -F "model_name=scrfd_10g" \
    -F "min_confidence=0.5"
  ```

### `/api/v1/insightface/verify`
- **Recommended models**: Full packages (`buffalo_l`, `buffalo_m`, `buffalo_s`)
- **Why**: Needs both detection and recognition for face comparison
- **Example**:
  ```bash
  curl -X POST /api/v1/insightface/verify \
    -F "file1=@face1.jpg" \
    -F "file2=@face2.jpg" \
    -F "model_name=buffalo_l" \
    -F "threshold=0.4"
  ```

### `/api/v1/insightface/recognize`
- **Recommended models**: Full packages (`buffalo_l`, `buffalo_m`, `buffalo_s`)
- **Why**: Needs detection + recognition + database matching
- **Example**:
  ```bash
  curl -X POST /api/v1/insightface/recognize \
    -F "file=@image.jpg" \
    -F "model_name=buffalo_l" \
    -F "threshold=0.35"
  ```

### `/api/v1/insightface/register`
- **Recommended models**: Full packages (`buffalo_l`, `buffalo_m`, `buffalo_s`)
- **Why**: Needs detection + recognition for feature extraction
- **Example**:
  ```bash
  curl -X POST /api/v1/insightface/register \
    -F "file=@face.jpg" \
    -F "name=John Doe" \
    -F "model_name=buffalo_l"
  ```

## Model Selection Guidelines

### For Detection Only
1. **High Accuracy**: Use `scrfd_10g`
2. **Balanced**: Use `scrfd_2.5g` 
3. **Speed Priority**: Use `scrfd_500m`

### For Full Face Recognition
1. **Production/High Accuracy**: Use `buffalo_l`
2. **Balanced Performance**: Use `buffalo_m`
3. **Mobile/Edge Devices**: Use `buffalo_s`
4. **Research/Maximum Accuracy**: Use `antelopev2`

## Performance Characteristics

### Detection Models (SCRFD)
- **scrfd_10g**: ~50ms/image on GPU, highest accuracy
- **scrfd_2.5g**: ~20ms/image on GPU, good accuracy
- **scrfd_500m**: ~10ms/image on GPU, acceptable accuracy

### Full Packages
- **buffalo_l**: ~100ms/image full pipeline
- **buffalo_m**: ~60ms/image full pipeline  
- **buffalo_s**: ~40ms/image full pipeline
- **antelopev2**: ~150ms/image full pipeline

## Memory Usage

### Detection Only
- SCRFD models: 20-100MB GPU memory

### Full Packages
- buffalo_s: ~300MB
- buffalo_m: ~800MB
- buffalo_l: ~1.5GB
- antelopev2: ~2GB+

## Model Switching Behavior

The API implements intelligent model management:

1. **Lazy Loading**: Models are loaded only when first used
2. **Auto-unloading**: When switching models, previous model is unloaded
3. **Model Sharing**: Same model instance is reused across multiple requests
4. **Smart Mapping**: Detection models automatically map to full packages when needed for recognition tasks