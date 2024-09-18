from pathlib import Path
import os
from datasets import load_dataset
from transformers import WhisperProcessor
import torch
import numpy as np
from torch.nn import functional as F
import onnxruntime as ort

os.environ["CUDA_VISIBLE_DEVICE"] = "3"
os.chdir(Path(__file__).parent)

# Select an audio file and read it:
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
audio_sample = ds[0]["audio"]

# Load the Whisper model in Hugging Face format:
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
processor.get_decoder_prompt_ids(task="transcribe", language="english")
input_features = processor(
    audio_sample["array"], sampling_rate=audio_sample["sampling_rate"], return_tensors="pt"
).input_features.half().numpy()

prompt_ids = processor(text="")["input_ids"][:-1]
print("Prompt ids:", prompt_ids)
text_features = torch.from_numpy(np.array(prompt_ids)).unsqueeze(0).to(torch.int64).numpy()
sess = ort.InferenceSession("whisper-large-v3-fp16-op10/model.onnx", providers=['CUDAExecutionProvider'])

while True:
    decoder_np = sess.run(None, {"input_features": input_features, "decoder_input_ids": text_features})[0]
    decoder_pt = torch.from_numpy(decoder_np)
    decoder_ids = torch.argmax(F.softmax(decoder_pt, dim=-1), dim=-1)
    new_token = decoder_ids.numpy()[0, -1]
    if new_token == 50257:
        break
    new_token = np.reshape(new_token, (1, 1))
    text_features = np.append(text_features, new_token, axis=-1)
    decode = processor.batch_decode(text_features)[0]
    print(decode)