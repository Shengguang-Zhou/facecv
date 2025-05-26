# Stream API Tests

This directory contains comprehensive tests for the FaceCV Stream API endpoints.

## Test Organization

### Test Files

1. **`test_stream_sources.py`** - Tests for `GET /api/v1/stream/sources`
   - Validates video source discovery
   - Checks response format and structure
   - Tests parameter handling

2. **`test_stream_process.py`** - Tests for `POST /api/v1/stream/process`
   - Local camera processing tests
   - RTSP stream processing tests
   - Parameter validation tests
   - Error handling tests

3. **`test_stream_comprehensive.py`** - Integration and comprehensive tests
   - Workflow integration tests
   - Performance baseline tests
   - Concurrent request handling
   - Error consistency tests

4. **`run_stream_tests.py`** - Test runner and orchestrator
   - Runs all test suites in organized manner
   - Provides detailed reporting
   - Includes smoke test functionality

## Running Tests

### Full Test Suite
```bash
python tests/api/stream/run_stream_tests.py
```

### Quick Smoke Test
```bash
python tests/api/stream/run_stream_tests.py --smoke
```

### Individual Test Files
```bash
# Run with pytest
pytest tests/api/stream/test_stream_sources.py -v
pytest tests/api/stream/test_stream_process.py -v
pytest tests/api/stream/test_stream_comprehensive.py -v

# Run as standalone scripts
python tests/api/stream/test_stream_sources.py
python tests/api/stream/test_stream_process.py
python tests/api/stream/test_stream_comprehensive.py
```

## Test Coverage

### Stream Sources API (`GET /api/v1/stream/sources`)
- ✅ Response structure validation
- ✅ Camera source detection
- ✅ RTSP examples format
- ✅ File format support
- ✅ Content type validation
- ✅ No parameters required

### Stream Processing API (`POST /api/v1/stream/process`)
- ✅ Local camera processing (index 0)
- ✅ RTSP stream processing
- ✅ Parameter validation (duration, skip_frames)
- ✅ Invalid source handling
- ✅ Minimal parameter sets
- ✅ Different skip_frames values
- ✅ Response format validation

### Integration Tests
- ✅ API server health checks
- ✅ Workflow integration (sources → process)
- ✅ Error handling consistency
- ✅ Response format consistency
- ✅ Performance baseline validation
- ✅ Concurrent request handling
- ✅ RTSP URL comprehensive testing

## Test Configuration

### Test Parameters
- **Local Camera**: Index 0 (configurable)
- **RTSP Stream**: `http://tdit.online:81/ai_check/TianduCV.git`
- **Server**: `http://localhost:7003`
- **Timeout**: 5-30 seconds depending on test type

### Expected Behaviors
- Camera tests may gracefully fail in CI environments
- RTSP tests handle network connectivity issues
- Short duration tests (1-5 seconds) for quick validation
- Skip frames (5-30) for faster processing during tests

## Test Results

### Success Criteria
- All critical functionality tests pass
- API endpoints return correct response formats
- Error handling is consistent and appropriate
- Performance meets baseline requirements

### Failure Handling
- Camera unavailability is handled gracefully
- Network issues with RTSP are acceptable
- Tests provide clear error messages and suggestions

## Dependencies

- `pytest` - Test framework
- `requests` - HTTP client for API testing
- `PIL` (Pillow) - Image processing
- `numpy` - Numerical operations

Install with:
```bash
pip install pytest requests pillow numpy
```

## Notes

- Tests assume the FaceCV API server is running on port 7003
- Camera tests may require actual camera hardware
- RTSP tests depend on network connectivity
- All tests are designed to be non-destructive and temporary