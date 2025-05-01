try:
    import debugpy
    debugpy.listen(("0.0.0.0", 5678))
    #debugpy.wait_for_client()
except:
    pass

import transformers
from datasets import load_dataset
import torch
import modelopt
from pathlib import Path
import onnx
import onnxruntime as ort
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import librosa

if __name__ == "__main__":
    # Load the model
    model_name = "openai/whisper-large-v3-turbo"
    
    dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
    
    whisper_model = WhisperForConditionalGeneration.from_pretrained(model_name,
                                                              torch_dtype=torch.float32,
                                                              device_map="cuda",
                                                              )
    processor = WhisperProcessor.from_pretrained(model_name)
    processor.tokenizer.set_prefix_tokens(language="italian", task="transcribe")
    prompt_ids = processor.get_decoder_prompt_ids()
    decoder_input_ids = [p[1] for p in prompt_ids]
    decoder_input_ids = torch.LongTensor(decoder_input_ids).unsqueeze(0)
    print("decoder_input_ids:", processor.decode(decoder_input_ids[0]))
    
    #Test sample
    sample = dataset[0]["audio"]
    inputs = processor(audio=sample["array"], 
                       sampling_rate=sample["sampling_rate"], 
                       return_tensors="pt",
                       truncation=True,
                       return_attention_mask=True)
    #inputs["decoder_input_ids"] = decoder_input_ids
    inputs = {k: v.to("cuda") for k, v in inputs.items()}   
    transription_ids = whisper_model.generate(**inputs)
    transcription = processor.batch_decode(transription_ids, skip_special_tokens=False)
    print(transcription[0])
    
    #Test sample
    sample, sr = librosa.load("test.wav", sr=16000)
    inputs = processor(audio=sample, 
                       sampling_rate=sr, 
                       return_tensors="pt",
                       truncation=True,
                       return_attention_mask=True)
    #inputs["decoder_input_ids"] = decoder_input_ids
    inputs = {k: v.to("cuda") for k, v in inputs.items()}   
    transription_ids = whisper_model.generate(**inputs)
    transcription = processor.batch_decode(transription_ids, skip_special_tokens=False)
    print(transcription[0])