export CUDA_HOME=/usr/local/cuda-12.0
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/nvidia/cublas/lib/

optimum-cli export onnx --model openai/whisper-large-v2 --optimize O4 --device cuda whisper-large-v2-optimized/