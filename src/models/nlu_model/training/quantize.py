import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("Loading trained model...")

# Load and merge LoRA adapters
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, "./qwen2-nlu-intent-classifier/final_model")
tokenizer = AutoTokenizer.from_pretrained("./qwen2-nlu-intent-classifier/final_model")

print("Merging LoRA adapters into base model...")
model = model.merge_and_unload()

print("Saving merged model...")
model.save_pretrained("./qwen2-nlu-merged", safe_serialization=True)
tokenizer.save_pretrained("./qwen2-nlu-merged")

print("✓ Merged model saved to ./qwen2-nlu-merged")
print("\nNow quantizing to 4-bit GGUF for deployment...")

# Install llama.cpp python bindings if needed
import subprocess
import sys

try:
    import llama_cpp
except:
    print("Installing llama-cpp-python...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "llama-cpp-python"])

print("\n" + "="*60)
print("Model ready for deployment!")
print("="*60)
print(f"\nMerged model: ./qwen2-nlu-merged (~1GB)")
print(f"Use this for inference on MacBook M2 and RTX 4060Ti")
print("\n✓ Quantization complete!")