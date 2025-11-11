import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

print("=" * 60)
print("STEP 4: Setting Up QLoRA")
print("=" * 60)

print("\n📚 QLoRA Concept:")
print("=" * 60)
print("""
QLoRA combines TWO techniques:

1. QUANTIZATION (Q):
    - Compress model from 16-bit → 4-bit
    - Reduces memory by 75%!
    - Model: 1GB → 250MB

2. LoRA (Low-Rank Adaptation):
    - Freeze original model weights
    - Add small "adapter" layers
    - Train only adapters (~1-2% of parameters)

Result: Train a 500M model like it's 10M!
""")

input("\nPress Enter to continue...")

# ============================================================
# PART 1: Quantization Configuration
# ============================================================

print("\n1. Setting up 4-bit Quantization")
print("=" * 60)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # Use 4-bit precision
    bnb_4bit_quant_type="nf4",              # NormalFloat4 - best for LLMs
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bfloat16
    bnb_4bit_use_double_quant=True,         # Double quantization
)

print("""
Configuration explained:

• load_in_4bit=True
└─> Each weight uses 4 bits instead of 16
    Example: 1.5 (16-bit) → 1011 (4-bit)

• bnb_4bit_quant_type="nf4" (NormalFloat4)
└─> Special 4-bit format optimized for neural networks
    Better than regular 4-bit integers!

• bnb_4bit_compute_dtype=torch.bfloat16
└─> During computation, convert back to bfloat16
    Storage: 4-bit, Computation: 16-bit

• bnb_4bit_use_double_quant=True
└─> Quantize the quantization parameters too!
    Saves even more memory (extra ~0.4 bits/param)
""")

input("\nPress Enter to load the model...")

# ============================================================
# PART 2: Load Model with Quantization
# ============================================================

print("\n2. Loading Qwen2-0.5B with 4-bit Quantization")
print("=" * 60)

model_name = "Qwen/Qwen2-0.5B"

print(f"Loading {model_name}...")
print("(This may take a minute - we're quantizing 500M parameters)")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",  # Automatically use available GPU/CPU
)

print("✓ Model loaded and quantized!")

# Check memory usage
model_size_bits = sum(p.numel() * 4 for p in model.parameters())  # 4 bits per param
model_size_gb = model_size_bits / 8 / 1e9  # Convert to GB

print(f"\n📊 Memory Usage:")
print(f"   Original (bfloat16): ~1.0 GB")
print(f"   Quantized (4-bit): ~{model_size_gb:.2f} GB")
print(f"   Savings: ~{(1.0 - model_size_gb) * 100:.0f}%")

input("\nPress Enter to add LoRA adapters...")

# ============================================================
# PART 3: Prepare Model for Training
# ============================================================

print("\n3. Preparing Model for k-bit Training")
print("=" * 60)

model = prepare_model_for_kbit_training(model)

print("""
What prepare_model_for_kbit_training does:

1. Enables gradient checkpointing
    └─> Trades computation for memory
    └─> Allows larger batch sizes

2. Freezes base model weights
    └─> No updates to the 4-bit weights
    └─> Only train adapters

3. Sets up proper dtype casting
    └─> Ensures gradients flow correctly through quantized layers
""")

input("\nPress Enter to configure LoRA...")

# ============================================================
# PART 4: LoRA Configuration
# ============================================================

print("\n4. Configuring LoRA Adapters")
print("=" * 60)

lora_config = LoraConfig(
    r=16,                    # Rank of LoRA matrices
    lora_alpha=32,           # Scaling factor
    target_modules=[         # Which layers get adapters
        "q_proj",           # Query projection
        "k_proj",           # Key projection
        "v_proj",           # Value projection
        "o_proj",           # Output projection
        "gate_proj",        # MLP gate
        "up_proj",          # MLP up
        "down_proj",        # MLP down
    ],
    lora_dropout=0.05,      # Dropout for regularization
    bias="none",            # Don't train bias terms
    task_type="CAUSAL_LM",  # Causal language modeling
)

print("""
LoRA Configuration explained:

• r=16 (Rank)
└─> Instead of updating full weight matrix W (large),
    add two small matrices: ΔW = B @ A
    where B is (d × 16) and A is (16 × k)
└─> Higher rank = more capacity (but more params)
└─> 16 is a sweet spot for most tasks

• lora_alpha=32
└─> Scaling factor = alpha / r = 32 / 16 = 2.0
└─> Controls how much the adapter affects output
└─> Usually set to 2×r

• target_modules
└─> Which layers get LoRA adapters
└─> We target: Attention (q,k,v,o) + MLP (gate,up,down)
└─> These are the most important layers to adapt

• lora_dropout=0.05
└─> 5% dropout on adapters for regularization
└─> Prevents overfitting
""")

input("\nPress Enter to add adapters to model...")

# ============================================================
# PART 5: Add LoRA to Model
# ============================================================

print("\n5. Adding LoRA Adapters to Model")
print("=" * 60)

model = get_peft_model(model, lora_config)

print("✓ LoRA adapters added!")

# ============================================================
# PART 6: Analyze Trainable Parameters
# ============================================================

print("\n6. Analyzing Parameters")
print("=" * 60)

trainable_params = 0
all_params = 0

for name, param in model.named_parameters():
    all_params += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
        # Show first few trainable layers
        if trainable_params < 1000000:  # First ~1M params
            print(f"   Trainable: {name:50s} {param.numel():>10,} params")

trainable_percentage = 100 * trainable_params / all_params

print(f"\n📊 Parameter Summary:")
print(f"   Total parameters:     {all_params:>12,}")
print(f"   Trainable parameters: {trainable_params:>12,}")
print(f"   Trainable:            {trainable_percentage:>11.2f}%")
print(f"\n   🎯 We're only training {trainable_percentage:.1f}% of the model!")

# ============================================================
# PART 7: Visualize What We Built
# ============================================================

print("\n7. Architecture Visualization")
print("=" * 60)
print("""
What we created:

┌────────────────────────────────────────┐
│   BASE MODEL (Frozen, 4-bit)           │
│   ┌──────────────────────────────┐    │
│   │  Attention Layer             │    │
│   │  • q_proj ←── LoRA adapter   │◄─── Trainable!
│   │  • k_proj ←── LoRA adapter   │◄─── Trainable!
│   │  • v_proj ←── LoRA adapter   │◄─── Trainable!
│   │  • o_proj ←── LoRA adapter   │◄─── Trainable!
│   └──────────────────────────────┘    │
│   ┌──────────────────────────────┐    │
│   │  MLP Layer                   │    │
│   │  • gate_proj ←── LoRA adapter│◄─── Trainable!
│   │  • up_proj ←── LoRA adapter  │◄─── Trainable!
│   │  • down_proj ←── LoRA adapter│◄─── Trainable!
│   └──────────────────────────────┘    │
│                                        │
│   ... 24 layers total ...             │
└────────────────────────────────────────┘

Total storage: ~250MB base + ~10MB adapters = ~260MB
""")

# ============================================================
# PART 8: Test the Model
# ============================================================

print("\n8. Quick Test (Before Training)")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(model_name)

test_prompt = """You are an expert voice assistant NLU system. Your task is to identify the user's intent.

Respond with ONLY a JSON object in this exact format:
{
"intent": "intent_name",
"confidence": 0.95
}

User utterance: play some jazz music"""

print(f"Test prompt:\n{test_prompt}\n")
print("Generating response (this will be random before training)...")

inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
response_only = response[len(test_prompt):]

print(f"Model output (before training):")
print(response_only)

print("\n" + "=" * 60)
print("QLoRA Setup Complete!")
print("=" * 60)

print("""
✅ What we accomplished:

1. Loaded Qwen2-0.5B in 4-bit (75% memory savings)
2. Added LoRA adapters (only ~1-2% trainable params)
3. Model ready for efficient training on H100s
4. Can train with large batch sizes thanks to low memory

💡 Key Insight:
    We're training <10M parameters to adapt a 500M model.
    This is like teaching a professor a new skill - you don't
    rewire their whole brain, just add specialized knowledge!

Next: Set up the training loop with our formatted dataset
""")