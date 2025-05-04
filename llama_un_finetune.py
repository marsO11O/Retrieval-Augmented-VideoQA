import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


dataset_path = "./olioli/dataset_processed.json"
with open(dataset_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

samples = []
if isinstance(dataset, dict) and "data" in dataset:
    for key, value in dataset["data"].items():
        samples.append(value)
else:
    samples = dataset


def create_prompt(sample):
    caption_lines = []
    for cap in sample["caption"]:
        caption_lines.append(f"[{cap['id']}] {cap['text']}")
    full_caption = "\n".join(caption_lines)
    
    prompt = (
        "Below is the video transcript :\n"
        f"{full_caption}\n\n"
        f"Question: {sample['question']}\n\n"
        "Based solely on the transcript provided above, answer the question using only the information explicitly stated in the transcript. Do not make any assumptions or inferences beyond what is directly mentioned.\n"
        "Determine whether the question can be confidently answered using only the transcript text: \n"
        "If the transcript provides sufficient evidence to fully answer the question, respond with ‘Answerability: 1’ \n"
        "If the transcript does not explicitly provide all the necessary details to fully answer the question—even if you indicate that the answer is unknown or incomplete-respond with ‘Answerability: 0’ \n\n"
        "Additionally, select the caption ID(s) that best support your determination. (Only select at most 3) \n\n"
        "Please ensure your answer strictly adheres to the transcript content without adding extra details. \n"
        "Answer format:\n"
        "Answerability: <0 or 1>\n"
        "Evidence: <comma-separated list of caption IDs>\n"
    )
    return prompt


model_name = "meta-llama/Llama-2-13b-chat-hf"
hf_token = ""
cache_dir = "/data/courses/2025/class_cse576spring2025_vgupt140/TBD/hf_cache"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    use_fast=False,
    token=hf_token
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    torch_dtype=torch.float16,
    device_map="auto",
    token=hf_token
)

if torch.cuda.is_available():
    model.to("cuda")
else:
    model.to("cpu")

generator = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_new_tokens=150
)


def choose_contiguous_range(numbers):
    if not numbers:
        return []
    numbers = sorted(set(numbers))
    groups = []
    current_group = [numbers[0]]
    for num in numbers[1:]:
        if num == current_group[-1] + 1:
            current_group.append(num)
        else:
            groups.append(current_group)
            current_group = [num]
    groups.append(current_group)

    max_group = max(groups, key=lambda g: len(g))
    return [max_group[0], max_group[-1]]


def get_prediction(prompt):
    full_output = generator(prompt, do_sample=False)[0]["generated_text"]

    generated_answer = full_output[len(prompt):].strip()

    ans_match = re.search(r"(?:Answerability|Answer):\s*([01])", generated_answer, re.IGNORECASE | re.DOTALL)

    evidence_match = re.search(r"Evidence:\s*(.*)", generated_answer, re.IGNORECASE | re.DOTALL)
    
    if ans_match:
        pred_answerability = int(ans_match.group(1))
    else:
        pred_answerability = 0
        
    if evidence_match:
        evidence_str = evidence_match.group(1).strip()
        if evidence_str.lower() in ["none", ""]:
            pred_evidence = []
        else:

            numbers = [int(token.strip()) for token in evidence_str.split(",") if token.strip().isdigit()]

            pred_evidence = choose_contiguous_range(numbers)
    else:
        pred_evidence = []
    
    return pred_answerability, pred_evidence, generated_answer


pred_answerabilities = []
gt_answerabilities = []
pred_evidences = []
gt_evidences = []

def expand_range(evidence_range):
    if evidence_range and len(evidence_range) == 2:
        return set(range(evidence_range[0], evidence_range[1] + 1))
    else:
        return set()

for sample in samples:
    prompt = create_prompt(sample)
    pred_ans, pred_evi, generated_answer = get_prediction(prompt)
    print("Model Output (generated part):\n", generated_answer)
    print("Predicted answerability:", pred_ans, "Predicted evidence:", pred_evi)
    print("Ground truth answerability:", sample["answerability"], "Ground truth evidence:", sample["evidence"])
    print("="*50)
    
    pred_answerabilities.append(pred_ans)
    gt_answerabilities.append(sample["answerability"])
    pred_evidences.append(expand_range(pred_evi))
    gt_evidences.append(expand_range(sample["evidence"]))

acc = accuracy_score(gt_answerabilities, pred_answerabilities)
binary_f1 = f1_score(gt_answerabilities, pred_answerabilities, average='binary')
print("Answerability Accuracy:", acc)
print("Answerability Binary F1:", binary_f1)
print("\nClassification Report for Answerability:")
print(classification_report(gt_answerabilities, pred_answerabilities))
print("Confusion Matrix for Answerability:")
print(confusion_matrix(gt_answerabilities, pred_answerabilities))

evidence_precisions = []
evidence_recalls = []
evidence_f1s = []

for pred_set, gt_set in zip(pred_evidences, gt_evidences):
    if len(pred_set) == 0 and len(gt_set) == 0:
        precision = recall = f1_val = 1.0
    elif len(pred_set) == 0 or len(gt_set) == 0:
        precision = recall = f1_val = 0.0
    else:
        intersection = pred_set.intersection(gt_set)
        precision = len(intersection) / len(pred_set) if len(pred_set) > 0 else 0.0
        recall = len(intersection) / len(gt_set) if len(gt_set) > 0 else 0.0
        if precision + recall == 0:
            f1_val = 0.0
        else:
            f1_val = 2 * precision * recall / (precision + recall)
    evidence_precisions.append(precision)
    evidence_recalls.append(recall)
    evidence_f1s.append(f1_val)

avg_evidence_precision = sum(evidence_precisions) / len(evidence_precisions)
avg_evidence_recall = sum(evidence_recalls) / len(evidence_recalls)
avg_evidence_f1 = sum(evidence_f1s) / len(evidence_f1s)

print("\nEvidence Metrics:")
print("Average Evidence Precision:", avg_evidence_precision)
print("Average Evidence Recall:", avg_evidence_recall)
print("Average Evidence F1:", avg_evidence_f1)