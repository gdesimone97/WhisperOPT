import torch
from optimum.pipelines import pipeline
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", device)
pipe = pipeline(
  "automatic-speech-recognition",
  model="openai/whisper-large-v2",
  chunk_length_s=30,
  device="cpu",
  accelerator="ort",
  cache_dir=r"S:\MIE CARTELLE\PROGRAMMAZIONE\GITHUB\WhisperOPT"
)
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = ds[0]["audio"]
prediction = pipe(sample.copy(), batch_size=8)["text"]