from datasets import load_dataset
import json

print("=" * 60)
print("STEP 2: Exploring MASSIVE Dataset (MTEB version)")
print("=" * 60)

print("\n1. Loading MASSIVE dataset (English)...")
print("   (This may take a minute on first run)")

# Load the new dataset
dataset = load_dataset("mteb/amazon_massive_intent", "en")

print(f"   ✓ Dataset loaded")

# Check what splits are available
print(f"\n2. Available splits:")
for split_name in dataset.keys():
    print(f"   - {split_name}: {len(dataset[split_name]):,} samples")

# Use the first available split to explore
split_name = list(dataset.keys())[0]
sample_data = dataset[split_name]

print(f"\n3. Dataset columns (features) in '{split_name}':")
for column in sample_data.column_names:
    print(f"   - {column}")

# Look at one example in detail
print("\n" + "=" * 60)
print("4. Let's examine ONE example in detail:")
print("=" * 60)

example = sample_data[0]
print(f"\nExample structure:")
for key, value in example.items():
    print(f"   {key}: {value}")
    print(f"      (type: {type(value).__name__})")

# Look at a few more examples
print("\n" + "=" * 60)
print("5. Let's look at 5 examples to understand patterns:")
print("=" * 60)

for i in range(5):
    example = sample_data[i]
    print(f"\n--- Example {i+1} ---")
    # Print all fields
    for key, value in example.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"{key}: {value[:100]}...")
        else:
            print(f"{key}: {value}")

print("\n" + "=" * 60)
print("6. Quick Statistics:")
print("=" * 60)

# Try to find label/intent field
if 'label' in sample_data.column_names:
    all_labels = [ex['label'] for ex in sample_data]
    unique_labels = set(all_labels)
    print(f"\nNumber of unique labels: {len(unique_labels)}")

    from collections import Counter
    label_counts = Counter(all_labels)
    print(f"\nMost common labels (top 10):")
    for label, count in label_counts.most_common(10):
        print(f"   - {label}: {count} examples")

print("\n" + "=" * 60)
print("Dataset exploration complete!")
print("Now we understand the structure and can format it properly.")
print("=" * 60)