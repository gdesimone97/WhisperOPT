import onnx

model = onnx.load("/user/gdesimone/code/WhisperOPT/whisper-large-v2-optimized/encoder_model.onnx")
print(model.opset_import)