import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

print("Loading model...")

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "./qwen2-nlu-intent-classifier/final_model")
tokenizer = AutoTokenizer.from_pretrained("./qwen2-nlu-intent-classifier/final_model")

model.eval()
print("✓ Model loaded\n")

def predict_intent(utterance):
    prompt = f"""<|im_start|>system
You are an expert voice assistant NLU system.<|im_end|>
<|im_start|>user
You are an expert voice assistant NLU system. Your task is to identify the user's intent.

Respond with ONLY a JSON object in this exact format:
{{
"intent": "intent_name",
"confidence": 0.95
}}

User utterance: {utterance}<|im_end|>
<|im_start|>assistant
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,  # Reduced
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

    # Extract first JSON
    try:
        start = response.find('{')
        end = response.find('}', start) + 1
        return json.loads(response[start:end])
    except:
        return {"raw_output": response}

# Test cases
test_utterances = [
    "wake me up at 7 am tomorrow",
    "play some jazz music",
    "what's the weather like in Seattle",
    "set a timer for 10 minutes",
    "send an email to John",
    "remind me to call mom",
]

print("Testing model:\n")
for utterance in test_utterances:
    result = predict_intent(utterance)
    print(f"Input:  {utterance}")
    print(f"Output: {json.dumps(result, indent=2)}\n")

print("✓ Testing complete!")