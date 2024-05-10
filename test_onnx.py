from datetime import datetime
from scipy.io import wavfile
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import pipeline, AutoTokenizer, AutoFeatureExtractor
from datasets import Audio, load_dataset
from pathlib import Path
import os

# Number of inferences for comparing timings
num_inferences = 4
save_dir = Path(__file__).parent.joinpath("whisper-medium-onnx") #complete here
inference_file_path =  Path(__file__).parent.joinpath("test.wav") #complete here

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = ds[0]["audio"]
inference_file = sample["array"]

tokenizer = AutoTokenizer.from_pretrained("openai/whisper-medium")
feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-medium")

r = feature_extractor(inference_file)
print(r, type(r))

r = tokenizer.decode([50258, 50363, 2755, 78, 50257])
print(r, type(r))