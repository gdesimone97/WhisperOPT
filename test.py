from datetime import datetime
from scipy.io import wavfile
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import pipeline, AutoTokenizer, AutoFeatureExtractor
from pathlib import Path

# Number of inferences for comparing timings
num_inferences = 4
save_dir = Path(__file__).parent.joinpath("whisper-large-no_conv") #complete here
inference_file_path =  Path(__file__).parent.joinpath("test.wav") #complete here
if not inference_file_path.exists():
    raise Exception(f"File: {inference_file_path} doesn't exist")
_, inference_file =  wavfile.read(inference_file_path)

print("Loading model...")
# Create pipeline based on quantized ONNX model
model = ORTModelForSpeechSeq2Seq.from_pretrained(save_dir)
tokenizer = AutoTokenizer.from_pretrained(save_dir)
feature_extractor = AutoFeatureExtractor.from_pretrained(save_dir)
cls_pipeline_onnx = pipeline("automatic-speech-recognition", model=model, tokenizer=tokenizer, feature_extractor=feature_extractor)

print("Starting inference...")
start_quantized = datetime.now()
for i in range(num_inferences):
    cls_pipeline_onnx(inference_file)
end_quantized = datetime.now()

quantized_inference_time = (end_quantized - start_quantized).total_seconds() / num_inferences
print(f"Quantized inference time: {quantized_inference_time}")