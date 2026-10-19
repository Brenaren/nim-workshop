import json
def read_jsonl (fname):
    obj = []
    with open(fname, 'rt') as f:
        st = f.readline()
        while st:
            obj.append(json.loads(st))
            st = f.readline()
    return obj
def write_jsonl(fname, json_objs):
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
    data = json.load(open(data_path, 'rt'))
    json_objs = []
    for k in data.keys():
        obj = data[k]
        prompt = form_question(obj)
        completion = obj['reasoning_required_pred']
        json_objs.append({"input": prompt, "output": completion})
    write_jsonl(output_path, json_objs)
    return json_objs
def main():
    test_json_objs = convert_to_jsonl("data/test_set.json", "pubmedqa_test.jsonl")
    train_json_objs = convert_to_jsonl("data/pqal_fold0/train_set.json", "pubmedqa_train.jsonl")
    dev_json_objs = convert_to_jsonl("data/pqal_fold0/dev_set.json", "pubmedqa_val.jsonl")
    return test_json_objs, train_json_objs, dev_json_objs
if __name__ == "__main__":
    main()