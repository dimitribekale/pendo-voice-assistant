import torch
from datasets import load_dataset, DatasetDict, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    #DataCollatorSpeechSeq2SeqWithPadding,
)
import evaluate
import re
from huggingface_hub import login
from config import Config


from typing import Any, Dict, List, Union
import torch

class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor: Any):
        self.processor = processor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Separate input features and labels
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad inputs and labels
        batch = self.processor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.pad(label_features, return_tensors="pt")

        # Replace padding with -100 for labels (ignored in loss computation)
        labels = labels_batch["input_ids"].masked_fill(labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id, -100)
        batch["labels"] = labels

        return batch


HUB_MODEL_ID = "bekalebendong/pendo-whisper-base-finetuned"
MODEL_CHECKPOINT = "openai/whisper-base"
DATASET_NAME = "superb"
CONFIG_NAME = "asr"
LANGUAGE_ABBR = "english"
TASK = "transcribe"
SAMPLING_RATE = 16000
HF_TOKEN = Config.HF_TOKEN_MODEL

# Check for GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def hf_login():
    print("Attempting to log in to HuggingFace Hub...")
    # You can pass your token directly or it will prompt you
    login(token=HF_TOKEN)  
    print("HuggingFace Hub login successful.")

hf_login()
# --- Load Feature Extractor, Tokenizer, and Processor ---
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_CHECKPOINT)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_CHECKPOINT, language=LANGUAGE_ABBR, task=TASK)
processor = WhisperProcessor.from_pretrained(MODEL_CHECKPOINT, language=LANGUAGE_ABBR, task=TASK)


common_voice = DatasetDict()
common_voice["train"] = load_dataset(DATASET_NAME, CONFIG_NAME, split="train", streaming=True)
common_voice["test"] = load_dataset(DATASET_NAME, CONFIG_NAME, split="validation", streaming=True)

print(f"Dataset loaded: {common_voice}")

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

# Characters to remove from the transcriptions
chars_to_remove_regex = r'[,\.?!\-;:\"“%‘”\']'

def remove_special_characters(batch):
    """Removes special characters. NOTE: The column is now 'text'."""
    batch["text"] = re.sub(chars_to_remove_regex, '', batch["text"]).lower()
    return batch

def prepare_dataset(batch):
    """
    Prepares a batch of audio data for the model.
    """
    audio = batch["audio"]

    # Compute log-Mel input features from the audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # Encode the target text to label ids
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch

print("Applying preprocessing to the dataset...")
common_voice = common_voice.map(remove_special_characters)
column_names = list(next(iter(common_voice.values())).features)
common_voice = common_voice.map(prepare_dataset, remove_columns=column_names)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


metric = evaluate.load("wer")

def compute_metrics(pred):
    """Computes Word Error Rate (WER) for the model's predictions."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with the pad token ID
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# --- Load Pre-trained Model ---
model = WhisperForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []


training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-base-finetuned-cv-en",
    per_device_train_batch_size=64,
    gradient_accumulation_steps=1, 
    learning_rate=15e-5,
    save_strategy="no",
    warmup_steps=500,
    max_steps=5000, 
    gradient_checkpointing=True, 
    bf16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=32,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=None,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    hub_model_id=HUB_MODEL_ID,
)


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# ====================================================================================
# 4. Start Training
# ====================================================================================

print("Starting the fine-tuning process...")
trainer.train()

print("Training complete. Pushing final model to the Hub...")
trainer.push_to_hub()

print(f"Model successfully pushed to: https://huggingface.co/{HUB_MODEL_ID}")