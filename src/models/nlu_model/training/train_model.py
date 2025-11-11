import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk


print("=" * 60)
print("STEP 5: Training the Model")
print("=" * 60)

# ============================================================
# PART 1: Load the Formatted Dataset
# ============================================================

print("\n1. Loading Formatted Dataset")
print("=" * 60)

train_dataset = load_from_disk("./data/massive_formatted_train")
val_dataset = load_from_disk("./data/massive_formatted_val")


print(f"✓ Training samples: {len(train_dataset):,}")
print(f"✓ Validation samples: {len(val_dataset):,}")

# Quick check
print(f"\nSample training example:")
print(f"  Prompt length: {len(train_dataset[0]['prompt'])} chars")
print(f"  Response length: {len(train_dataset[0]['response'])} chars")

input("\nPress Enter to continue...")

# ============================================================
# PART 2: Load Model and Tokenizer with QLoRA
# ============================================================

print("\n2. Loading Model with QLoRA Setup")
print("=" * 60)

model_name = "Qwen/Qwen2-0.5B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set padding token (required for training)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"  Set pad_token to eos_token: '{tokenizer.eos_token}'")

print("\nLoading model with 4-bit quantization...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": torch.cuda.current_device()},
)

print("✓ Model loaded in 4-bit")

# Prepare for training
model = prepare_model_for_kbit_training(model)

# Add LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
print("✓ LoRA adapters added")

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"✓ Trainable: {trainable_params:,} / {all_params:,} ({100*trainable_params/all_params:.2f}%)")

input("\nPress Enter to set up data processing...")

# ============================================================
# PART 3: Data Collator - Format Data for Training
# ============================================================

print("\n3. Setting Up Data Collator")
print("=" * 60)

print("""
Data Collator's job:
1. Combine prompt + response into one sequence
2. Tokenize the text
3. Create labels (mask the prompt, only train on response)
4. Handle batching and padding
""")

def formatting_func(example):
    """
    Combine prompt and response into training format

    Format for Qwen2:
    <|im_start|>system
    You are a helpful assistant<|im_end|>
    <|im_start|>user
    {prompt}<|im_end|>
    <|im_start|>assistant
    {response}<|im_end|>
    """

    # Qwen2 chat template format
    text = f"""<|im_start|>system
You are an expert voice assistant NLU system.<|im_end|>
<|im_start|>user
{example['prompt']}<|im_end|>
<|im_start|>assistant
{example['response']}<|im_end|>"""

    return {"text": text}

# Apply formatting
print("Applying chat template to datasets...")
train_dataset = train_dataset.map(formatting_func)
val_dataset = val_dataset.map(formatting_func)

print("✓ Chat template applied")

# Show example
print(f"\nFormatted example (first 200 chars):")
print(train_dataset[0]['text'][:200] + "...")

def tokenize_function(example):
    """
    Tokenize and create labels with masking
    """
    # Get the formatted text
    text = example['text']

    # Tokenize
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding=False,  # Padding will be done by collator
    )

    # Create labels (mask instruction part)
    labels = tokenized["input_ids"].copy()

    # Find assistant response start
    assistant_start = text.find("<|im_start|>assistant\n")
    if assistant_start != -1:
        prefix = text[:assistant_start + len("<|im_start|>assistant\n")]
        prefix_tokens = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        prefix_len = len(prefix_tokens)

        # Mask prefix
        labels[:prefix_len] = [-100] * prefix_len

    tokenized["labels"] = labels
    return tokenized

print("Tokenizing datasets...")
train_dataset = train_dataset.map(tokenize_function, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(tokenize_function, remove_columns=val_dataset.column_names)
print("✓ Datasets tokenized")

# Use built-in data collator for padding
from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

print("\n💡 Key Insight:")
print("   We mask the instruction part (set to -100)")
print("   Model only learns to generate the JSON response")
print("   This is critical for instruction tuning!")

input("\nPress Enter to configure training...")

# ============================================================
# PART 4: Training Configuration
# ============================================================

print("\n4. Configuring Training Arguments")
print("=" * 60)

# Create output directory
output_dir = "./qwen2-nlu-intent-classifier"
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,

    # Larger batch size for single H100 80GB
    per_device_train_batch_size=16,      # Increased from 4
    per_device_eval_batch_size=32,       # Increased from 8
    gradient_accumulation_steps=2,        # Reduced from 4
    # Effective batch = 16 * 2 = 32 (still good!)

    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,

    optim="paged_adamw_8bit",
    weight_decay=0.01,
    max_grad_norm=1.0,

    bf16=True,

    logging_steps=10,
    logging_dir=f"{output_dir}/logs",

    eval_strategy="steps",
    eval_steps=100,

    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",

    dataloader_num_workers=4,
    group_by_length=False,

    report_to="tensorboard",
)

print("Training configuration:")
print(f"  Total epochs: {training_args.num_train_epochs}")
print(f"  Batch size per device: {training_args.per_device_train_batch_size}")
print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps} (per GPU)")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Optimizer: {training_args.optim}")

# Calculate training steps
steps_per_epoch = len(train_dataset) // (training_args.per_device_train_batch_size *
training_args.gradient_accumulation_steps * 4)  # 4 GPUs
total_steps = steps_per_epoch * training_args.num_train_epochs

print(f"\n📊 Training Overview:")
print(f"  Steps per epoch: ~{steps_per_epoch}")
print(f"  Total training steps: ~{total_steps}")
print(f"  Evaluation every: {training_args.eval_steps} steps")
print(f"  Estimated time: ~{total_steps * 2 / 60:.1f} minutes (rough estimate)")

input("\nPress Enter to create trainer...")

# ============================================================
# PART 5: Create Trainer
# ============================================================

print("\n5. Creating Trainer")
print("=" * 60)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

print("✓ Trainer created")
print("\nThe Trainer will handle:")
print("  • Multi-GPU distribution (across your 4 H100s)")
print("  • Gradient accumulation")
print("  • Mixed precision training")
print("  • Checkpointing")
print("  • Logging to TensorBoard")
print("  • Evaluation during training")

input("\nPress Enter to start training...")

# ============================================================
# PART 6: Train!
# ============================================================

print("\n6. Starting Training")
print("=" * 60)
print("\n🚀 Training starting now!")
print("=" * 60)

# Train the model
trainer.train()

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)

# ============================================================
# PART 7: Save the Model
# ============================================================

print("\n7. Saving Model")
print("=" * 60)

# Save the final model
trainer.save_model(f"{output_dir}/final_model")
tokenizer.save_pretrained(f"{output_dir}/final_model")

print(f"✓ Model saved to: {output_dir}/final_model")

# Save only LoRA adapters (much smaller!)
model.save_pretrained(f"{output_dir}/lora_adapters")
print(f"✓ LoRA adapters saved to: {output_dir}/lora_adapters")

adapter_size = sum(
    os.path.getsize(os.path.join(f"{output_dir}/lora_adapters", f))
    for f in os.listdir(f"{output_dir}/lora_adapters")
    if os.path.isfile(os.path.join(f"{output_dir}/lora_adapters", f))
)
print(f"  Adapter size: ~{adapter_size / 1e6:.1f} MB")

print("\n" + "=" * 60)
print("🎉 ALL DONE!")
print("=" * 60)

print("""
What you accomplished:
✅ Trained a 500M parameter model
✅ Using only ~10M trainable parameters (QLoRA)
✅ On your 4 H100 GPUs
✅ With proper instruction tuning
✅ Model saved and ready for inference!

Next steps:
1. Test the model on new examples
2. Evaluate on test set
3. Deploy for inference
4. (Optional) Add entity extraction

Check TensorBoard for training curves:
tensorboard --logdir ./qwen2-nlu-intent-classifier/logs
""")