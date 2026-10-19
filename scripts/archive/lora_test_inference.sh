#!/bin/bash
MODEL="/debug/nemo/mistral-7b.nemo"
PATH_TO_TRAINED_MODEL=/debug/results/checkpoints/megatron_gpt_peft_lora_tuning.nemo
# if argument passed use $1 for filename, else use default
TEST_DS="[/debug/utils/pubmedqa/pubmedqa_test_small.jsonl]"
TEST_NAMES="[pubmedqa]"
OUTPUT_PREFIX=test
TP_SIZE=1
PP_SIZE=1

# Output files written here
cd /debug/nemofw_notebooks

# Run Without LoRA
python3 /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_generate.py \
    model.restore_from_path=${MODEL} \
    trainer.devices=1 \
    model.tensor_model_parallel_size=${TP_SIZE} \
    model.pipeline_model_parallel_size=${PP_SIZE} \
    model.data.test_ds.file_names=${TEST_DS} \
    model.data.test_ds.names=${TEST_NAMES} \
    model.global_batch_size=32 \
    model.micro_batch_size=4 \
    model.data.test_ds.tokens_to_generate=20 \
    model.megatron_amp_O2=True \
    inference.greedy=True \
    model.data.test_ds.output_file_path_prefix=no_lora \
    model.data.test_ds.write_predictions_to_file=True \
    model.data.test_ds.label_key='output' \
    model.data.test_ds.add_eos=True \
    model.data.test_ds.add_sep=False \
    model.data.test_ds.add_bos=False \
    model.data.test_ds.truncation_field="input" \
    model.data.test_ds.prompt_template="\{input\} \{output\}" \

# Run With LoRA - added model.peft.restore_from_path
python3 /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_generate.py \
    model.restore_from_path=${MODEL} \
    model.peft.restore_from_path=${PATH_TO_TRAINED_MODEL} \
    trainer.devices=1 \
    model.tensor_model_parallel_size=${TP_SIZE} \
    model.pipeline_model_parallel_size=${PP_SIZE} \
    model.data.test_ds.file_names=${TEST_DS} \
    model.data.test_ds.names=${TEST_NAMES} \
    model.global_batch_size=32 \
    model.micro_batch_size=4 \
    model.data.test_ds.tokens_to_generate=20 \
    model.megatron_amp_O2=True \
    inference.greedy=True \
    model.data.test_ds.output_file_path_prefix=with_lora \
    model.data.test_ds.write_predictions_to_file=True