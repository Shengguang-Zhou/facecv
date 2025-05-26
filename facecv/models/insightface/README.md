# InsightFace Mock Implementation

This directory contains a mock implementation of the InsightFace face recognition system for FaceCV.

## Purpose

These mock implementations serve as compatibility layers for the InsightFace functionality in FaceCV. They allow the codebase to function without requiring the actual InsightFace library to be installed, which can be useful in the following scenarios:

- Development environments where InsightFace dependencies are not available
- Testing environments where real face recognition is not needed
- CI/CD pipelines where installing the full InsightFace library would be resource-intensive

## Implementation Details

The mock implementation:

- Provides the same API as the real InsightFace implementation
- Returns realistic but fake data for all operations
- Logs clear warnings when used to indicate it's a mock implementation
- Maintains type compatibility with the real implementation

## Files

- `__init__.py` - Package initialization and imports
- `real_recognizer.py` - Mock implementation of RealInsightFaceRecognizer
- `onnx_recognizer.py` - Mock implementation of ONNXFaceRecognizer
- `arcface_recognizer.py` - Mock implementation of ArcFaceRecognizer
- `arcface_models.py` - Mock implementation of ArcFace model loading

## Usage

The mock implementation is used automatically when the real InsightFace library is not available. To use the real implementation, ensure the InsightFace library is installed:

```bash
pip install insightface
```

## Limitations

The mock implementation:

- Does not perform actual face recognition
- Returns predefined mock data instead of real analysis
- Should not be used in production environments where real face recognition is required
