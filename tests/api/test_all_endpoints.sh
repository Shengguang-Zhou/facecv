#!/bin/bash
# Comprehensive API endpoint testing script

BASE_URL="http://localhost:7000"
TEST_IMAGE="/home/a/PycharmProjects/EurekCV/dataset/faces/trump1.jpeg"

echo "=== Testing FaceCV API Endpoints on Port 7000 ==="
echo

# 1. Health Check
echo "1. Testing Health Check Endpoint"
curl -s "$BASE_URL/health" | jq .
echo

# 2. API Documentation
echo "2. Checking API Documentation"
curl -s -o /dev/null -w "Documentation URL: $BASE_URL/docs - Status: %{http_code}\n" "$BASE_URL/docs"
echo

# 3. InsightFace Endpoints
echo "3. Testing InsightFace API Endpoints"
echo "   a. Register Face"
curl -s -X POST "$BASE_URL/api/v1/face_recognition_insightface/faces/register" \
  -F "name=Trump" \
  -F "file=@$TEST_IMAGE" \
  -F "department=Politics" \
  -F "employee_id=001" | jq .
echo

echo "   b. List Faces"
curl -s "$BASE_URL/api/v1/face_recognition_insightface/faces" | jq .
echo

echo "   c. Get Face Count"
curl -s "$BASE_URL/api/v1/face_recognition_insightface/faces/count" | jq .
echo

echo "   d. Recognize Face"
curl -s -X POST "$BASE_URL/api/v1/face_recognition_insightface/faces/recognize" \
  -F "file=@$TEST_IMAGE" \
  -F "threshold=0.6" | jq .
echo

echo "   e. Verify Faces"
curl -s -X POST "$BASE_URL/api/v1/face_recognition_insightface/faces/verify" \
  -F "file1=@$TEST_IMAGE" \
  -F "file2=@/home/a/PycharmProjects/EurekCV/dataset/faces/trump2.jpeg" \
  -F "threshold=0.6" | jq .
echo

echo "   f. Offline Batch Registration"
curl -s -X POST "$BASE_URL/api/v1/face_recognition_insightface/faces/offline" \
  -F "directory_path=/home/a/PycharmProjects/EurekCV/dataset/faces" \
  -F "quality_threshold=0.5" | jq .
echo

# 4. DeepFace Endpoints
echo "4. Testing DeepFace API Endpoints"
echo "   a. Analyze Face"
curl -s -X POST "$BASE_URL/api/v1/face_recognition_deepface/analyze/" \
  -F "file=@$TEST_IMAGE" | jq .
echo

echo "   b. Verify Faces"
curl -s -X POST "$BASE_URL/api/v1/face_recognition_deepface/verify/" \
  -F "file1=@$TEST_IMAGE" \
  -F "file2=@/home/a/PycharmProjects/EurekCV/dataset/faces/trump3.jpeg" | jq .
echo

echo "   c. Register Face"
curl -s -X POST "$BASE_URL/api/v1/face_recognition_deepface/faces/" \
  -F "file=@$TEST_IMAGE" \
  -F "name=Trump" \
  -F "metadata={\"role\":\"president\"}" | jq .
echo

echo "   d. List Faces"
curl -s "$BASE_URL/api/v1/face_recognition_deepface/faces/" | jq .
echo

echo "   e. Recognition"
curl -s -X POST "$BASE_URL/api/v1/face_recognition_deepface/recognition" \
  -F "file=@$TEST_IMAGE" | jq .
echo

echo "   f. Health Check"
curl -s "$BASE_URL/api/v1/face_recognition_deepface/health" | jq .
echo

# 5. Video Processing (if video file exists)
echo "5. Testing Video Processing Endpoints"
if [ -f "test_video.mp4" ]; then
    echo "   a. InsightFace Video Face Extraction"
    curl -s -X POST "$BASE_URL/api/v1/face_recognition_insightface/video_face/" \
      -F "file=@test_video.mp4" \
      -F "method=uniform" \
      -F "count=5" | jq .
    
    echo "   b. DeepFace Video Face Extraction"
    curl -s -X POST "$BASE_URL/api/v1/face_recognition_deepface/video_face/" \
      -F "file=@test_video.mp4" | jq .
else
    echo "   Skipping video tests (no test_video.mp4 found)"
fi
echo

# 6. Stream Endpoints (just check availability)
echo "6. Testing Stream Endpoints"
echo "   a. InsightFace Webcam Stream"
curl -s -o /dev/null -w "Stream URL: $BASE_URL/api/v1/face_recognition_insightface/recognize/webcam/stream - Status: %{http_code}\n" \
  "$BASE_URL/api/v1/face_recognition_insightface/recognize/webcam/stream?source=0"

echo "   b. DeepFace Webcam Stream"
curl -s -o /dev/null -w "Stream URL: $BASE_URL/api/v1/face_recognition_deepface/recognize/webcam/stream - Status: %{http_code}\n" \
  "$BASE_URL/api/v1/face_recognition_deepface/recognize/webcam/stream"
echo

echo "=== Testing Complete ==="