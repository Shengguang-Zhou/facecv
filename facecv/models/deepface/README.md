# DeepFace Mock Implementation

This directory contains a mock implementation of the DeepFace face recognition system for FaceCV.

## Purpose

These mock implementations serve as compatibility layers for the DeepFace functionality in FaceCV. They allow the codebase to function without requiring the actual DeepFace library to be installed, which can be useful in the following scenarios:

- Development environments where DeepFace dependencies are not available
- Testing environments where real face recognition is not needed
- CI/CD pipelines where installing the full DeepFace library would be resource-intensive

## Implementation Details

The mock implementation:

- Provides the same API as the real DeepFace implementation
- Returns realistic but fake data for all operations
- Logs clear warnings when used to indicate it's a mock implementation
- Maintains type compatibility with the real implementation

## Files

- `__init__.py` - Package initialization and imports
- `recognizer.py` - Mock implementation of DeepFaceRecognizer
- `embedding.py` - Mock implementation of face embedding functions
- `verification.py` - Mock implementation of face verification functions
- `analysis.py` - Mock implementation of face analysis functions

## Usage

The mock implementation is used automatically when the real DeepFace library is not available. To use the real implementation, ensure the DeepFace library is installed:

```bash
pip install deepface
```

## Limitations

The mock implementation:

- Does not perform actual face recognition
- Returns predefined mock data instead of real analysis
- Should not be used in production environments where real face recognition is required
