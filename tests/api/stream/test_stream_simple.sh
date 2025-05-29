#!/bin/bash

# Test stream processing API

echo "Testing stream processing without webhook (should process and return summary)..."
echo

# Test 1: Process default camera for 5 seconds
echo "Test 1: Process default camera"
curl -X POST "http://localhost:7003/api/v1/stream/process?source=0&skip_frames=2&show_preview=false" \
  --max-time 10

echo
echo

# Test 2: Process with webhook URL
echo "Test 2: Process with webhook URL (will run continuously)"
echo "Press Ctrl+C to stop..."
curl -X POST "http://localhost:7003/api/v1/stream/process?source=0&webhook_url=http://localhost:8080/webhook&skip_frames=2&show_preview=false"