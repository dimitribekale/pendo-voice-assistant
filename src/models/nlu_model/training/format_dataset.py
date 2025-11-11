from datasets import load_dataset
import json

print("=" * 60)
print("STEP 3: Formatting Dataset for Training")
print("=" * 60)

# Load dataset
print("\n1. Loading dataset...")
dataset = load_dataset("mteb/amazon_massive_intent", "en")
print("   ✓ Loaded")

print("\n2. Understanding Instruction Tuning Format")
print("=" * 60)
print("""
Instruction tuning teaches the model to follow instructions.

Format:
┌─────────────────────────────────────┐
│ INSTRUCTION (what to do)            │
│ "Extract the intent from..."        │
├─────────────────────────────────────┤
│ INPUT (user utterance)              │
│ "wake me up at nine am on friday"   │
├─────────────────────────────────────┤
│ RESPONSE (what we want the model    │
│ to learn to generate)               │
│ {"intent": "alarm_set", ...}        │
└─────────────────────────────────────┘

The model learns: Given instruction + input → generate response
""")

input("\nPress Enter to continue...")

print("\n3. Creating the formatting function")
print("=" * 60)

def format_for_training(example):
    """
    Convert raw example into instruction-tuning format

    Input example:
    {
        "text": "wake me up at nine am on friday",
        "label": "alarm_set"
    }

    Output format:
    {
        "prompt": "Instruction + user input",
        "response": "JSON with intent"
    }
    """

    # The instruction tells the model what task to perform
    instruction = """You are an expert voice assistant NLU system. Your task is to identify the user's intent.

Respond with ONLY a JSON object in this exact format:
{
"intent": "intent_name",
"confidence": 0.95
}

User utterance: """

    # Build the complete prompt
    prompt = instruction + example["text"]

    # Build the expected response (what we want the model to learn)
    response = json.dumps({
        "intent": example["label"],
        "confidence": 1.0  # Ground truth has max confidence
    }, indent=2)

    return {
        "prompt": prompt,
        "response": response,
        "original_text": example["text"],
        "original_label": example["label"]
    }

print("\n4. Let's format a few examples and see the transformation:")
print("=" * 60)

# Format 3 examples to show the transformation
for i in range(3):
    raw_example = dataset["train"][i]
    formatted = format_for_training(raw_example)

    print(f"\n--- Example {i+1} ---")
    print(f"ORIGINAL:")
    print(f"  Text: {raw_example['text']}")
    print(f"  Label: {raw_example['label']}")

    print(f"\nFORMATTED FOR TRAINING:")
    print(f"\n[PROMPT - What the model sees]")
    print(formatted["prompt"])

    print(f"\n[RESPONSE - What the model should generate]")
    print(formatted["response"])

    print("\n" + "-" * 60)

    if i < 2:
        input("Press Enter for next example...")

print("\n5. Applying to entire dataset...")
print("=" * 60)

# Apply formatting to all splits
formatted_train = dataset["train"].map(format_for_training)
formatted_val = dataset["validation"].map(format_for_training)
formatted_test = dataset["test"].map(format_for_training)

print(f"✓ Training set formatted: {len(formatted_train):,} examples")
print(f"✓ Validation set formatted: {len(formatted_val):,} examples")
print(f"✓ Test set formatted: {len(formatted_test):,} examples")

print("\n6. Saving formatted dataset...")
print("=" * 60)

# Save to disk for training
formatted_train.save_to_disk("./data/massive_formatted_train")
formatted_val.save_to_disk("./data/massive_formatted_val")
formatted_test.save_to_disk("./data/massive_formatted_test")

print("✓ Saved to ./data/massive_formatted_train")
print("✓ Saved to ./data/massive_formatted_val")
print("✓ Saved to ./data/massive_formatted_test")

print("\n7. Quick verification - loading back one example:")
print("=" * 60)

from datasets import load_from_disk
loaded = load_from_disk("./data/massive_formatted_train")
example = loaded[0]

print(f"Loaded example has fields: {list(example.keys())}")
print(f"\nPrompt preview (first 100 chars):")
print(f"  {example['prompt'][:100]}...")
print(f"\nResponse:")
print(f"  {example['response']}")

print("\n" + "=" * 60)
print("SUCCESS! Dataset is formatted and ready for training!")
print("=" * 60)

print("""
📚 What we learned:
1. Instruction tuning = teaching models to follow instructions
2. Format: Instruction + Input → Response
3. We want JSON output for structured NLU
4. All 11k+ examples are now properly formatted!

Next step: We'll set up the model with QLoRA for efficient training
""")