## Evaluation of PEFT model
# Script downloads autotokenizer, so need this login
huggingface-cli login --token $HF_TOKEN

# The number of total GPUs available
GPU_COUNT=2

# Change this to the nemo model you want to use
MODEL="./models/llama-8b-instruct.nemo" #"gemma_2b_base_v1.1/pytorch-2b-pt.nemo"

# This is the dataset you are using. 
# You can either use the "pubmedqa" dataset for yes/no use case, 
# or the " att_dataset" for the summarization use case.
DATASET_DIRECTORY="pubmedqa"

# The filename will match whichever peft scheme was used during training
PATH_TO_TRAINED_MODEL="./nemo_experiments/megatron_gpt_peft_lora_tuning/checkpoints/megatron_gpt_peft_lora_tuning.nemo"


# The test dataset
TEST_DS="${DATASET_DIRECTORY}/test_set.jsonl"
TEST_NAMES="${DATASET_DIRECTORY}"

# This is the prefix, including the path and filename prefix, for the accuracy file output
# This will be combined with TEST_NAMES to create the file ./results/peft_results_test_att_dataset_inputs_preds_labels.jsonl
OUTPUT_PREFIX="/results/peft_results"

# Change the number of output tokens depending on the use case
TOKENS_TO_GENERATE=1024

# This is the tensor parallel size (splitting tensors among GPUs horizontally)
TP_SIZE=1

# This is the pipeline parallel size (splitting layers among GPUs vertically)
PP_SIZE=1

# Execute
python /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_generate.py \
    model.restore_from_path=${MODEL} \
    model.peft.restore_from_path=${PATH_TO_TRAINED_MODEL} \
    trainer.devices=${GPU_COUNT} \
    model.micro_batch_size=32 \
    model.global_batch_size=64 \
    model.data.test_ds.file_names=[${TEST_DS}] \
    model.data.test_ds.names=[${TEST_NAMES}] \
    model.data.test_ds.micro_batch_size=32 \
    model.data.test_ds.global_batch_size=64 \
    model.data.test_ds.tokens_to_generate=${TOKENS_TO_GENERATE} \
    model.tensor_model_parallel_size=${TP_SIZE} \
    trainer.precision=bf16 \
    model.megatron_amp_O2=True \
    model.pipeline_model_parallel_size=${PP_SIZE} \
    inference.greedy=True \
    model.data.test_ds.output_file_path_prefix=${OUTPUT_PREFIX} \
    model.answer_only_loss=True \
    model.data.test_ds.write_predictions_to_file=True
