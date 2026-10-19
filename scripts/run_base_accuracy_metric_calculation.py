import json
#change base/peft results file beginning 
OUTPUT_FILE = "./results/base_results_test_pubmedqa_inputs_preds_labels.jsonl"
answers = []
with open(OUTPUT_FILE,'rt') as f:
    st = f.readline()
    while st:
        answers.append(json.loads(st))
        st = f.readline()

data_test = json.load(open("./pubmedqa/data/test_set.json",'rt'))

results = {}
sample_id = list(data_test.keys())

for i, key in enumerate(sample_id):
    answer = answers[i]['pred']
    if 'yes' in answer:
        results[key] = 'yes'
    elif 'no' in answer:
        results[key] = 'no'
    elif 'maybe' in answer:
        results[key] = 'maybe'
    else:
        print("Malformed answer: ", answer)
        results[key] = 'maybe'

# Save the results
FILENAME="./scripts/pubmedqa-base-accuracy-results.json"
with(open(FILENAME, "w")) as f:
    json.dump(results, f)
