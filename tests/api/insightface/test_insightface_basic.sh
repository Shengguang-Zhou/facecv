#!/bin/bash

# InsightFace API Basic Test Script
# Tests the main endpoints with local test image

BASE_URL="http://localhost:7003/api/v1/insightface"
TEST_IMAGE="/home/a/PycharmProjects/facecv/test_images/test_face.jpg"

echo "=== InsightFace API Test ==="
echo

# 1. Health Check
echo "1. Testing Health Check..."
curl -X GET "${BASE_URL}/health" -s | python -m json.tool
echo

# 2. Face Detection
echo "2. Testing Face Detection..."
curl -X POST "${BASE_URL}/detect" \
  -F "file=@${TEST_IMAGE}" \
  -F "model=buffalo_s" \
  -F "min_confidence=0.5" \
  -s | python -m json.tool | head -20
echo

# 3. Face Registration
echo "3. Testing Face Registration..."
TIMESTAMP=$(date +%s)
curl -X POST "${BASE_URL}/register" \
  -F "file=@${TEST_IMAGE}" \
  -F "name=Test User ${TIMESTAMP}" \
  -F "department=Engineering" \
  -F "employee_id=EMP${TIMESTAMP}" \
  -s | python -m json.tool
echo

# 4. List Faces
echo "4. Testing List Faces..."
curl -X GET "${BASE_URL}/faces?limit=5" -s | python -m json.tool | head -30
echo

# 5. Face Recognition
echo "5. Testing Face Recognition..."
curl -X POST "${BASE_URL}/recognize" \
  -F "file=@${TEST_IMAGE}" \
  -F "threshold=0.3" \
  -s | python -m json.tool | head -30
echo

echo "=== Test Complete ==="