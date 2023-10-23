import torch
from optimum.pipelines import pipeline
from datasets import load_dataset
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
curr_dir = Path(__file__).parent
print("Device: ", device)
pipe = pipeline(
  "automatic-speech-recognition",
  model="openai/whisper-large-v2",
  chunk_length_s=30,
  device="cpu",
  accelerator="ort",
  cache_dir=curr_dir
)
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = ds[0]["audio"]
prediction = pipe(sample.copy(), batch_size=8)["text"]