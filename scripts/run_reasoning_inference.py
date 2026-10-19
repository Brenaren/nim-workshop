from megatron.core.inference.common_inference_params import CommonInferenceParams
import nemo.lightning as nl
from nemo.collections.llm import api
import torch

strategy = nl.MegatronStrategy(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    sequence_parallel=False,
    setup_optimizers=False,
)

trainer = nl.Trainer(
    accelerator="gpu",
    devices=1,
    num_nodes=1,
    strategy=strategy,
    plugins=nl.MegatronMixedPrecision(
        precision="bf16-mixed",
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
    ),
)

# Note, prompt formatting is a big deal.  We did LORA FT with a specific input format, see what happens as you change the format .. 
# no formatting - this is out of distribution relative to our fine tuning data set
# formatting with proper special tokens + detailed thinking on / off

# Feel free to modify and add other questions ... 

prompts = [
    "How many r's are in the word 'strawberry'?",
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\ndetailed thinking off<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHow many r's in strawberry?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\ndetailed thinking on<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHow many r's in strawberry?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "Which number is bigger? 10.119 or 10.19?",
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\ndetailed thinking off<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhich number is bigger 10.119 or 10.19?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\ndetailed thinking on<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhich number is bigger 10.119 or 10.19?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    ]


if __name__ == "__main__":

    adapter_checkpoint = "/workspace/models/llama31_reasoning_lora"  # a folder that contains "weights" and "context" subfolders
    results = api.generate(
        path=adapter_checkpoint,
        prompts=prompts,
        trainer=trainer,
        inference_params=CommonInferenceParams(temperature=0.1, top_k=10, num_tokens_to_generate=512),
        text_only=True,
        output_path="/workspace/output/inference_results.jsonl"
    
    )

