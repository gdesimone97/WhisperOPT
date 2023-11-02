from pathlib import Path
from functools import partial
from transformers import AutoTokenizer
from optimum.onnxruntime import (
    AutoQuantizationConfig,
    ORTModelForSpeechSeq2Seq,
    ORTQuantizer
)

# Configure base model and save directory for compressed model
model_id = "openai/whisper-large-v2"
save_dir = Path(__file__).parent.joinpath("whisper-large-tensorrt")
cache_dir = Path(__file__).parent.joinpath(f'{model_id.split("/")[1]}')

# Export model in ONNX
model = ORTModelForSpeechSeq2Seq.from_pretrained(model_id, export=True, cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(model_id) #tokenaizer
quantizer = ORTQuantizer.from_pretrained(model)
qconfig = AutoQuantizationConfig.tensorrt(per_channel=False) #quantization config
print("Model loaded")
model_dir = model.model_save_dir

# Run quantization for all ONNX files of exported model
onnx_models = list(Path(model_dir).glob("*.onnx"))
print(onnx_models)
quantizers = [ORTQuantizer.from_pretrained(model_dir, file_name=onnx_model) for onnx_model in onnx_models]

#Calibration dataset
def preprocess_fn(ex, tokenizer):
    return tokenizer(ex["sentence"])

calibration_dataset = quantizer.get_calibration_dataset(
    "hf-internal-testing/librispeech_asr_dummy",
    preprocess_function=partial(preprocess_fn, tokenizer=tokenizer),
    num_samples=250,
    dataset_split="train",
)

calibration_config = AutoCalibrationConfig.minmax(calibration_dataset) #calibration config
ranges = quantizer.fit(
    dataset=calibration_dataset,
    calibration_config=calibration_config,
    operators_to_quantize=qconfig.operators_to_quantize,
)

for quantizer in quantizers:
    # Apply dynamic quantization and save the resulting model
    quantizer.quantize(save_dir=save_dir, quantization_config=qconfig)