import evaluate
import json

OUTPUT_FILE = "./data/peft_results_test_pubmedqa_inputs_preds_labels.jsonl"

# Load the jsonl file
jsonl_data = []
with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
    jsonl_data = [json.loads(line) for line in f]

# Define the predictions and reference sentences
predictions = [data['pred'] for data in jsonl_data]
references = [[data['label']] for data in jsonl_data]

# Load and compute the BLEU evaluation metric
bleu = evaluate.load("bleu")
bleu_results = bleu.compute(predictions=predictions, references=references)
print(bleu_results)

# Load and compute ROUGE evaluation metric
rouge = evaluate.load('rouge')
rouge_results = rouge.compute(predictions=predictions, references=references)
print(rouge_results)