import json
import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

os.environ["ACCELERATE_USE_DISTRIBUTED_TENSOR"] = "0"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def create_transcript(captions):
    captions = sorted(captions, key=lambda x: x["id"])
    transcript_lines = [f"Caption {cap['id']}: {cap['text'][:100]}" for cap in captions]
    return "\n".join(transcript_lines)

def load_data_from_json(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data["data"].values())

class VideoQADataset(Dataset):
    """
      Question: <question>
      Transcript:
      Caption 0: <text>
      Caption 1: <text>
      ...
      Answerability:

      <answerability>
      Evidence: <evidence indices>
    """
    def __init__(self, data, tokenizer, max_length=256, margin=0):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.margin = margin
        self.samples = []

        for item in self.data:
            question = item["question"]
            captions = item.get("caption", [])
            evidence_list = item.get("evidence", [])
            


            if isinstance(item["answerability"], list):
                answerability = str(item["answerability"][0])
            else:
                answerability = str(item["answerability"])
            evidence = ",".join(map(str, item["evidence"]))
            
            if len(captions) > 0 and len(evidence_list) > 0:
                e_min = min(evidence_list)
                e_max = max(evidence_list)

                sub_start = max(0, e_min - self.margin)
                sub_end = min(len(captions), e_max + self.margin + 1)
                sub_captions = captions[sub_start:sub_end]
            else:
                sub_captions = captions


            transcript = create_transcript(sub_captions)


            prompt_text = (
                f"Question: {question}\nTranscript:\n{transcript}\nAnswerability:"
            )
            answer_text = f" {answerability}\nEvidence: {evidence}"

            self.samples.append((prompt_text, answer_text))

    
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        prompt_text, answer_text = self.samples[idx]

        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer_text, add_special_tokens=False)

        total_len = len(prompt_ids) + len(answer_ids)
        if total_len > self.max_length:
            ans_len = len(answer_ids)
            allow_prompt_len = self.max_length - ans_len
            if allow_prompt_len < 0:

                answer_ids = answer_ids[: self.max_length]
                prompt_ids = []
            else:

                prompt_ids = prompt_ids[-allow_prompt_len:]

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids

        # Padding
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [pad_id] * pad_len
            labels += [-100] * pad_len

        attention_mask = [1 if tid != pad_id else 0 for tid in input_ids]


        num_label_tokens = sum(l != -100 for l in labels)
        print(f"[DEBUG] idx={idx}, total_len={len(input_ids)}, label_tokens={num_label_tokens}")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

def load_data_from_json(json_file):

    with open(json_file, 'r') as f:
        data = json.load(f)
    return list(data["data"].values())

def main():
    
    use_manual_split = True
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    cache_dir = "/data/courses/2025/class_cse576spring2025_vgupt140/TBD/hf_cache"

    auth_token = ""
    

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, token=auth_token)

    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    

    if use_manual_split:
        train_data = load_data_from_json("./olioli/processed_data/train_dataset.json")
        valid_data = load_data_from_json("./olioli/processed_data/test_dataset.json")
    else:
        full_data = load_data_from_json("./olioli/data_01.json")
        train_data, valid_data = train_test_split(full_data, test_size=0.2, random_state=42)
    
    train_data_small = train_data[:50]
    train_data_small = VideoQADataset(train_data_small, tokenizer, max_length=256)
    valid_dataset = VideoQADataset(valid_data, tokenizer, max_length=256)
    

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        cache_dir=cache_dir, 
        device_map="auto", 
        token=auth_token,
        #low_cpu_mem_usage=True,
        torch_dtype="auto",
    )

    #model.gradient_checkpointing_enable()
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        evaluation_strategy="epoch",
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        fp16=False,
        bf16=False,
        learning_rate=1e-5,
        max_grad_norm=1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data_small,
        eval_dataset=valid_dataset,
    )
    
    trainer.train()

if __name__ == "__main__":
    main()
