try:
    import debugpy
    debugpy.listen(("0.0.0.0", 5678))
    #debugpy.wait_for_client()
except:
    pass

import transformers
from datasets import load_dataset
import torch
import modelopt.torch.quantization as mtq
from pathlib import Path
import onnx
import onnxruntime as ort
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from tqdm import tqdm
import numpy as np
import shutil

MAX_ITER = 100
NUM_STEPS = 0

class Decoder(torch.nn.Module):
    def __init__(self, decoder, proj_out):
        super().__init__()
        self.decoder = decoder
        self.proj_out = proj_out
    
    def forward(self, input_ids, encoder_hidden_states):
        decoder_out = self.decoder(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states)
        logits = self.proj_out(decoder_out.last_hidden_state)
        return logits

def forward_encoder(model):
    global NUM_STEPS
    batch_size = 1
    dataset_iter = dataset.iter(batch_size=batch_size)
    bar = tqdm(total=MAX_ITER, desc="Calibrating", unit="batch")
    for i, sample in enumerate(dataset_iter):
        sample = sample["audio"]
        if i >= MAX_ITER:
            break
        sr = [s["sampling_rate"] for s in sample]
        sr_0 = sr[0]
        assert np.all(np.array(sr) == sr_0), "All samples must have the same sampling rate"
        audio = [s["array"] for s in sample]
        inputs = processor(audio=audio, 
                           sampling_rate=sr_0, 
                           return_tensors="pt",
                           truncation=True,
                           return_attention_mask=True)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        out = model(**inputs)
        last_hidden_state = out.last_hidden_state
        torch.save(
            last_hidden_state, 
            out_tensors_path.joinpath(f"last_hidden_{i}.pt")
        )
        bar.update(1)
        NUM_STEPS = i
        
def forward_decoder(model):
    bar = tqdm(total=MAX_ITER, desc="Calibrating", unit="batch")
    decoder_prompt_ids = processor(text="", return_tensors="pt")["input_ids"][..., :-1]
    eos = processor.tokenizer.eos_token_id
    new_token = None
    for i in range(NUM_STEPS):
        last_hidden_state = torch.load(
            out_tensors_path.joinpath(f"last_hidden_{i}.pt")
        ).to("cuda")
        while new_token is None or (new_token != eos and j < 400):
            j = 0
            inputs = {}
            inputs["input_ids"] = decoder_prompt_ids
            inputs["encoder_hidden_states"] = last_hidden_state
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            logits = model(**inputs)
            new_token = torch.argmax(logits, dim=-1)[:, -1].unsqueeze(0)
            decoder_prompt_ids = torch.cat([decoder_prompt_ids.to("cuda"), new_token], dim=-1)
            out_text = processor.batch_decode(decoder_prompt_ids, skip_special_tokens=False)
            j += 1
        print(f"Decoded text: {out_text}")
        bar.update(1)

if __name__ == "__main__":
    # Load the model
    fname_encoder = "whisper_large_v3_turbo_encoder.onnx"
    fname_decoder = "whisper_large_v3_turbo_decoder.onnx"
    out_tensors_path = Path("tensors")
    out_tensors_path.mkdir(parents=True, exist_ok=True)
    
    out_tmp_encoder = Path("onnx_model_encoder_tmp").joinpath(fname_encoder)
    out_tmp_decoder = Path("onnx_model_decoder_tmp").joinpath(fname_decoder)
    out_tmp_encoder.parent.mkdir(parents=True, exist_ok=True)
    out_tmp_decoder.parent.mkdir(parents=True, exist_ok=True)
    
    out_encoder = Path("onnx_model_encoder").joinpath(fname_encoder)
    out_encoder.parent.mkdir(parents=True, exist_ok=True)
    
    out_decoder = Path("onnx_model_decoder").joinpath(fname_decoder)
    out_decoder.parent.mkdir(parents=True, exist_ok=True)
    
    model_name = "openai/whisper-large-v3"
    
    dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    
    whisper_model = WhisperForConditionalGeneration.from_pretrained(model_name,
                                                              torch_dtype=torch.float32,
                                                              device_map="cuda",
                                                              )
    encoder = whisper_model.get_encoder()
    whisper_model = WhisperForConditionalGeneration.from_pretrained(model_name,
                                                              torch_dtype=torch.float32,
                                                              device_map="cuda",
                                                              )
    decoder = whisper_model.get_decoder()
    decoder = Decoder(decoder, whisper_model.proj_out).to("cuda")
    processor = WhisperProcessor.from_pretrained(model_name)
    processor.tokenizer.set_prefix_tokens(language="en", task="transcribe")
    config = mtq.INT8_DEFAULT_CFG
    dummy = torch.randn(1, 128, 3000).to("cuda")
    
    encoder = mtq.quantize(encoder, config, forward_encoder)
    decoder = mtq.quantize(decoder, config, forward_decoder)
    shutil.rmtree(out_tensors_path)
    
    dynamic_axes = {
        "mel": {2: "sequence"},
    }
    torch.onnx.export(encoder, 
                      (dummy,), 
                      f=str(out_tmp_encoder),
                      input_names=["mel"],
                      output_names=["embeddings"],
                      dynamic_axes=dynamic_axes,
                      )
    onnx_model = onnx.load_model(str(out_tmp_encoder))
    onnx.save_model(onnx_model, 
                    out_encoder, 
                    save_as_external_data=True, 
                    all_tensors_to_one_file=True, 
                    location=Path(fname_encoder).stem + "_weights.onnx", 
                    convert_attribute=False)
    
    shutil.rmtree(out_tmp_encoder.parent)
    
    #encoder output: Bx1500x1280 (Float)
    #deconder input: BxDecoder_Ids (LongTensor)
    #                Bx1500x1280 (Float)

    decoder_prompt_ids = processor.get_decoder_prompt_ids()
    decoder_prompt_ids = [p[1] for p in decoder_prompt_ids]
    decoder_prompt_ids = torch.LongTensor(decoder_prompt_ids).unsqueeze(0).to("cuda")
    enc_output = encoder(dummy).last_hidden_state.to("cuda")   
    dummy = {
        "input_ids": decoder_prompt_ids, 
        "encoder_hidden_states": enc_output,
    }
    dynamic_axes = {
        "prompt_ids": {1: "seq_len"},
    }
    torch.onnx.export(decoder, 
                      kwargs=dummy, 
                      f=str(out_tmp_decoder),
                      input_names=["prompt_ids", "embedding"],
                      output_names=["logits"],
                      dynamic_axes=dynamic_axes,
                      )
    onnx_model = onnx.load_model(str(out_tmp_decoder))
    onnx.save_model(onnx_model, 
                    out_decoder, 
                    save_as_external_data=True, 
                    all_tensors_to_one_file=True, 
                    location=Path(fname_decoder).stem + "_weights.onnx", 
                    convert_attribute=False)
    
    shutil.rmtree(out_tmp_decoder.parent)