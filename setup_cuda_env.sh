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
