import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 60)
print("STEP 1: Loading and Exploring Qwen2-0.5B")
print("=" * 60)

# Model we'll use as our base
model_name = "Qwen/Qwen2-0.5B"

print(f"\n1. Loading tokenizer from {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"   ✓ Tokenizer loaded")
print(f"   - Vocabulary size: {len(tokenizer)}")
print(f"   - Special tokens: {tokenizer.special_tokens_map}")

print(f"\n2. Loading model (this may take a minute)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,  # Use bfloat16 to save memory
    device_map="cpu",             # Load on CPU for now (change to "cuda" if you want GPU)
)

print(f"   ✓ Model loaded")

# Analyze the model architecture
print("\n3. Model Architecture:")
print(f"   - Model type: {model.config.model_type}")
print(f"   - Number of layers: {model.config.num_hidden_layers}")
print(f"   - Hidden size: {model.config.hidden_size}")
print(f"   - Attention heads: {model.config.num_attention_heads}")
print(f"   - Max position embeddings: {model.config.max_position_embeddings}")

# Calculate model size
total_params = sum(p.numel() for p in model.parameters())
print(f"\n4. Model Size:")
print(f"   - Total parameters: {total_params:,}")
print(f"   - Size in memory (bfloat16): ~{total_params * 2 / 1e9:.2f} GB")

# Test the model with a simple prompt
print("\n5. Testing base model (before fine-tuning):")
test_prompt = "Hello, I am"
inputs = tokenizer(test_prompt, return_tensors="pt")

print(f"   Input: '{test_prompt}'")
print(f"   Generating...")

with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=20,
        temperature=0.7,
        do_sample=True,
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"   Output: '{generated_text}'")

print("\n" + "=" * 60)
print("Exploration complete! The model is ready for fine-tuning.")
print("=" * 60)