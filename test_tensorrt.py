from datetime import datetime
from scipy.io import wavfile
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import pipeline, AutoTokenizer, AutoFeatureExtractor
from datasets import Audio, load_dataset
from pathlib import Path
import os

#os.environ["CUDA_PATH"] = "/usr/local/cuda"
#os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.0/lib64"


# Number of inferences for comparing timings
num_inferences = 4
save_dir = Path(__file__).parent.joinpath("whisper-large-tensorrt") #complete here
inference_file_path =  Path(__file__).parent.joinpath("test.wav") #complete here

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = ds[0]["audio"]
inference_file = sample["array"]

print("Loading model...")

provider_options = {
    "trt_engine_cache_enable": True,
    "trt_engine_cache_path": "tmp/trt_cache_example"
}

# Create pipeline based on quantized ONNX model
model = ORTModelForSpeechSeq2Seq.from_pretrained(save_dir,
                                                 use_cache=False,
                                                 provider="TensorrtExecutionProvider",
                                                 provider_options=provider_options)
tokenizer = AutoTokenizer.from_pretrained(save_dir)
feature_extractor = AutoFeatureExtractor.from_pretrained(save_dir)
cls_pipeline_onnx = pipeline("automatic-speech-recognition", model=model, tokenizer=tokenizer, feature_extractor=feature_extractor)

print("Starting inference...")
start_quantized = datetime.now()
for i in range(num_inferences):
    res = cls_pipeline_onnx(inference_file)
    print(res)
end_quantized = datetime.now()

quantized_inference_time = (end_quantized - start_quantized).total_seconds() / num_inferences
print(f"Quantized inference time: {quantized_inference_time}")