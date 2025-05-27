#!/bin/bash
# Setup CUDA libraries for ONNX Runtime GPU support

echo "Setting up CUDA libraries for ONNX Runtime..."

# Check if CUDA is already installed
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA driver detected"
    nvidia-smi
else
    echo "⚠ NVIDIA driver not detected. GPU support will not be available."
    exit 0
fi

# Check CUDA version
if [ -d "/usr/local/cuda" ]; then
    echo "✓ CUDA installation found at /usr/local/cuda"
    ls -la /usr/local/cuda/lib64/libcudart.so* 2>/dev/null || echo "⚠ CUDA runtime libraries not found"
else
    echo "⚠ CUDA not found at /usr/local/cuda"
fi

# Add CUDA to library path
echo "Adding CUDA libraries to LD_LIBRARY_PATH..."
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Create a script to set environment variables
cat > setup_cuda_env.sh << 'EOF'
#!/bin/bash
# Source this file to set CUDA environment variables

export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

# For ONNX Runtime
export ORT_CUDA_PROVIDER_OPTIONS="device_id=0;arena_extend_strategy=kNextPowerOfTwo;gpu_mem_limit=2147483648;cudnn_conv_algo_search=EXHAUSTIVE;do_copy_in_default_stream=True"

echo "CUDA environment variables set:"
echo "  CUDA_HOME=$CUDA_HOME"
echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
EOF

chmod +x setup_cuda_env.sh

# Install ONNX Runtime GPU if not already installed
echo ""
echo "Checking ONNX Runtime GPU installation..."
python -c "import onnxruntime; print(f'ONNX Runtime version: {onnxruntime.__version__}'); print(f'Available providers: {onnxruntime.get_available_providers()}')" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "Installing ONNX Runtime GPU..."
    pip install onnxruntime-gpu
else
    # Check if GPU provider is available
    python -c "import onnxruntime; providers = onnxruntime.get_available_providers(); exit(0 if 'CUDAExecutionProvider' in providers else 1)"
    if [ $? -ne 0 ]; then
        echo "⚠ CUDAExecutionProvider not available. You may need to:"
        echo "  1. Install CUDA 11.x or 12.x"
        echo "  2. Install cuDNN"
        echo "  3. Reinstall onnxruntime-gpu: pip install --force-reinstall onnxruntime-gpu"
    else
        echo "✓ ONNX Runtime GPU support is available"
    fi
fi

echo ""
echo "To use GPU support, run: source setup_cuda_env.sh"
echo "Then start your application."