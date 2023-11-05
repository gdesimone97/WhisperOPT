from pathlib import Path

from optimum.onnxruntime import (
    AutoOptimizationConfig, ORTOptimizer, ORTModelForSpeechSeq2Seq, OptimizationConfig
)

curr_dir = Path(__file__).parent
model_id = "openai/whisper-large-v2"
model_name = model_id.split("/")[1]
save_dir = curr_dir.joinpath((model_name + "-optimized"))

model = ORTModelForSpeechSeq2Seq.from_pretrained(model_id, export=True, 
                                                 cache_dir=curr_dir.joinpath((model_name + "-cache")))

print("Model loaded")
optimizer = ORTOptimizer.from_pretrained(model)
optimization_config = AutoOptimizationConfig.O4()
optimizer.optimize(save_dir=save_dir, optimization_config=optimization_config)
optimized_model = ORTModelForSpeechSeq2Seq.from_pretrained(save_dir)
print("All ok")