from datetime import datetime
from scipy.io import wavfile
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import pipeline, AutoTokenizer, AutoFeatureExtractor
from datasets import Audio, load_dataset
from pathlib import Path
import numpy as np

def pcm2float(sig, dtype='float32'):
    """Convert PCM signal to floating point with a range from -1 to 1.
    Use dtype='float32' for single precision.
    Parameters
    ----------
    sig : array_like
        Input array, must have integral type.
    dtype : data type, optional
        Desired (floating point) data type.
    Returns
    -------
    numpy.ndarray
        Normalized floating point data.
    See Also
    --------
    float2pcm, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max

# Number of inferences for comparing timings
num_inferences = 4
save_dir = Path(__file__).parent.joinpath("whisper-large-no_conv") #complete here

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = ds[0]["audio"]
inference_file = sample["array"]

print("Loading model...")

# Create pipeline based on quantized ONNX model
model = ORTModelForSpeechSeq2Seq.from_pretrained(save_dir,
                                                 provider="CPUExecutionProvider",
                                                 )
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

inference_file_path =  Path(__file__).parent.joinpath("test.wav") #complete here
_, inference_file = wavfile.read(inference_file_path)
inference_file = pcm2float(inference_file)

print("Starting inference local file...")
start_quantized = datetime.now()
for i in range(num_inferences):
    res = cls_pipeline_onnx(inference_file)
    print(res)
end_quantized = datetime.now()

quantized_inference_time = (end_quantized - start_quantized).total_seconds() / num_inferences
print(f"Quantized inference time: {quantized_inference_time}")