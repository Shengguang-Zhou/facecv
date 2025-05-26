#!/bin/bash
# Test script for new InsightFace API endpoints

BASE_URL="http://localhost:8000/api/v1/face_recognition_insightface"

echo "=== Testing InsightFace Video Face Extraction ==="
echo "Testing POST /video_face/"
# Note: Requires a video file
curl -X POST "$BASE_URL/video_face/" \
  -F "file=@test_video.mp4" \
  -F "method=uniform" \
  -F "count=10" \
  -F "quality_threshold=0.7" | jq .

echo -e "\n=== Testing InsightFace Webcam Stream Recognition ==="
echo "Testing GET /recognize/webcam/stream"
# This will start a SSE stream - press Ctrl+C to stop
curl -N "$BASE_URL/recognize/webcam/stream?source=0&threshold=0.6&fps=10"

echo -e "\n=== Testing InsightFace Offline Batch Registration ==="
echo "Testing POST /faces/offline"
# Using the test faces directory
curl -X POST "$BASE_URL/faces/offline" \
  -F "directory_path=/home/a/PycharmProjects/EurekCV/dataset/faces" \
  -F "quality_threshold=0.7" | jq .