# Adapt for Launchpad ! (and any other machine for that matter!)

import subprocess
import argparse
import os.path
import pandas as pd
from io import StringIO


def run_genai_perf(
    nickname="",
    model="mistralai/mixtral-8x22b-instruct-v0-1",
    endpoint_type="chat",
    service_kind="openai",
    synthetic_input_tokens_mean=700,
    concurrency=1000,
    output_tokens_mean=300,
    tokenizer="hf-internal-testing/llama-tokenizer",
    url="http://0.0.0.0:8000",
    max_threads=256,
):

    profile_export_file = f"/opt/profiling_reports/{nickname}_profile_export.json"
    cmd = [
        "genai-perf",
        "--model", model,
        "--endpoint-type", endpoint_type,
        "--service-kind", service_kind,
        "--streaming",
        "--synthetic-input-tokens-mean", str(synthetic_input_tokens_mean),
        "--concurrency", str(concurrency),
        "--output-tokens-mean", str(output_tokens_mean),
        "--output-tokens-stddev", "0",
        "--tokenizer", tokenizer,
        "--url", url,
        "--profile-export-file", os.path.basename(profile_export_file),
        "--artifact-dir", os.path.dirname(profile_export_file),
        # not a random seed. give deterministic results
        "--random-seed", "1234",
        "--extra-inputs", "ignore_eos:true",
        "--measurement-interval", "40000",
        "--",
        "-v",
        "--max-threads", str(max_threads),
    ]

    print(f"Running GenAI-Perf with command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Read the JSON profile data
    #with open(profile_export_file, "r") as f:
    #    profile_data = json.load(f)

    # Read the CSV performance data
    csv_file = profile_export_file.replace(".json","_genai_perf.csv")
    csv_content = None
    with open(csv_file, "r") as f:
        csv_content = f.read()

    return transform_csv(csv_content)

def transform_csv(csv_content):
    # Splitting the two parts of the input CSV
    part1, part2 = csv_content.split("\n\n")
    
    # Read each part into a DataFrame
    df1 = pd.read_csv(StringIO(part1))
    df2 = pd.read_csv(StringIO(part2))
    
    # Creating new DataFrame with columns as desired
    new_df = pd.DataFrame()
    
    # Transform df1 to the new format
    for _, row in df1.iterrows():
        metric_name = row['Metric']
        for col in df1.columns[1:]:
            new_df[f'{metric_name} {col}'] = [row[col]]
    
    # Adding df2 to the new format
    for _, row in df2.iterrows():
        new_df[row['Metric']] = [row['Value']]
    
    # Convert the new_df to CSV string format
    output_csv = new_df.to_csv(index=False)
    return output_csv

def run_concurrency_sweep(args, concurrencies=[1, 10, 25, 50, 100, 150, 200, 300, 400, 500]):
    all_csv_results = []
    try:
        for concurrency in concurrencies:
        #for concurrency in [1, 10]:
            conc_nickname = f"{args['nickname']}_c{concurrency}"
            csv_results = run_genai_perf(**{
                **args,
                "concurrency": concurrency,
                "nickname": conc_nickname,
                "synthetic_input_tokens_mean": isl,
                "output_tokens_mean": osl,
            })

            #combine CSVs
            if len(all_csv_results) == 0:
                all_csv_results.append(csv_results.split("\n")[0]+",Concurrency,Name")

            all_csv_results.append(csv_results.split("\n")[1] + f",{concurrency},{conc_nickname}")
    except subprocess.CalledProcessError as e:
        print(f"Error occured after {len(all_csv_results)} tests: {e.stderr}")

    output_csv_name = f"/opt/profiling_reports/{args['nickname']}_genai_perf_combined.csv"
    with open(output_csv_name, "w") as f:
        f.write("\n".join(all_csv_results))
    print(f"Wrote results to {output_csv_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GenAI-Perf")
    parser.add_argument("--nickname")
    parser.add_argument("--model", default="mistralai/mixtral-8x22b-instruct-v0-1")
    parser.add_argument("--endpoint-type", default="chat")
    parser.add_argument("--service-kind", default="openai")
    parser.add_argument("--synthetic-input-tokens-mean", "--isl", type=int, default=700)
    #parser.add_argument("--concurrency", type=int, default=1000)
    parser.add_argument("--output-tokens-mean", "--osl", type=int, default=300)
    parser.add_argument("--tokenizer", default="hf-internal-testing/llama-tokenizer")
    parser.add_argument("--url", default="http://0.0.0.0:8000")
    parser.add_argument("--max-threads", type=int, default=256)

    args = parser.parse_args()
    for isl in [1024, 2048, 4096, 8192, 16384]:
        for osl in [1024, 2048, 4096, 8192, 16384]:
            nickname_with_isl_osl = f"{args.nickname}_isl{isl}_osl{osl}"
            run_concurrency_sweep({
                **vars(args),
                "nickname": nickname_with_isl_osl,
                "synthetic_input_tokens_mean": isl,
                "output_tokens_mean": osl,
            }, concurrencies=[1, 10, 25, 50, 100, 150, 200])