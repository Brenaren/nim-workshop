import json

def write_jsonl(fname, json_objs):
    print(f"writing out jsonl to {fname}")
    with open(fname, 'wt') as f:
        for o in json_objs:
            f.write(json.dumps(o)+"\n")

def form_question(obj):
    st = ""
    st += f"QUESTION:{obj['QUESTION']}\n"
    st += "CONTEXT: "
    for i, label in enumerate(obj['LABELS']):
        st += f"{obj['CONTEXTS'][i]}\n"
    st += f"TARGET: the answer to the question given the context is (yes|no|maybe): "
    return st

def convert_to_jsonl(data_path, output_path):
    print(f"Processing {data_path}")
    data = json.load(open(data_path, 'rt'))
    json_objs = []
    for k in data.keys():
        obj = data[k]
        prompt = form_question(obj)
        completion = obj['reasoning_required_pred']
        json_objs.append({"input": prompt, "output": completion})
    write_jsonl(output_path, json_objs)
    return json_objs

test_json_objs = convert_to_jsonl("./pubmedqa/data/test_set.json", "test_set.jsonl")
train_json_objs = convert_to_jsonl("./pubmedqa/data/pqal_fold0/train_set.json", "train_set.jsonl")
dev_json_objs = convert_to_jsonl("./pubmedqa/data/pqal_fold0/dev_set.json", "val_set.jsonl")