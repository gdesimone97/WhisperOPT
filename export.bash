export LD_LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/nvidia/cublas/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.8/dist-packages/nvidia/cufft/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.8/dist-packages/nvidia/cuda_runtime/lib/

# optimum-cli export onnx --model openai/whisper-medium --optimize O4 --device cuda whisper-large-v2-optimized/
# optimum-cli export onnx --model openai/whisper-medium --fp16 --opset 13 --device cuda whisper-medium-onnx/
optimum-cli export onnx  --model openai/whisper-large-v3 --monolith --dtype fp16 --opset 10 --device cuda whisper-large-v3-fp16-op10/