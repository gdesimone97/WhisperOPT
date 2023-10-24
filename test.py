from datetime import datetime
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import pipeline, AutoTokenizer, AutoFeatureExtractor
from pathlib import Path

# Number of inferences for comparing timings
num_inferences = 4
save_dir = Path(__file__).parent.joinpath("whisper-large") #complete here
inference_file =  Path(__file__).parent.joinpath("test.wav") #complete here
if not inference_file.exists():
    raise Exception(f"File: {inference_file} doesn't exist")

print("Starting")
# Create pipeline based on quantized ONNX model
model = ORTModelForSpeechSeq2Seq.from_pretrained(save_dir)
tokenizer = AutoTokenizer.from_pretrained(save_dir)
feature_extractor = AutoFeatureExtractor.from_pretrained(save_dir)
cls_pipeline_onnx = pipeline("automatic-speech-recognition", model=model, tokenizer=tokenizer, feature_extractor=feature_extractor, accelerator="ort")

start_quantized = datetime.now()
for i in range(num_inferences):
    cls_pipeline_onnx(inference_file)
end_quantized = datetime.now()

quantized_inference_time = (end_quantized - start_quantized).total_seconds() / num_inferences
print(f"Quantized inference time: {quantized_inference_time}")