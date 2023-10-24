from pathlib import Path
from optimum.onnxruntime import (
    AutoQuantizationConfig,
    ORTModelForSpeechSeq2Seq,
    ORTQuantizer
)

# Configure base model and save directory for compressed model
model_id = "openai/whisper-large-v2"
save_dir = Path(__file__).parent.joinpath("whisper-large-tensorrt")

# Export model in ONNX
model = ORTModelForSpeechSeq2Seq.from_pretrained(model_id, export=True)
print("Model loaded")
model_dir = model.model_save_dir

# Run quantization for all ONNX files of exported model
onnx_models = list(Path(model_dir).glob("*.onnx"))
print(onnx_models)
quantizers = [ORTQuantizer.from_pretrained(model_dir, file_name=onnx_model) for onnx_model in onnx_models]

qconfig = AutoQuantizationConfig.avx512(is_static=False, per_channel=False)

for quantizer in quantizers:
    # Apply dynamic quantization and save the resulting model
    quantizer.quantize(save_dir=save_dir, quantization_config=qconfig)