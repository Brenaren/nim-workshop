# This will be run inside docker container.  Paths are based on proper volume mappings in docker run 
# within notebook.

# This is the nemo model we are finetuning
# Change this to match the model you want to finetune
MODEL="/debug/nemo/mistral-7b.nemo"


# These are the training datasets (in our case we only have one)
# These are the validation datasets (in our case we only have one)
# These are the test datasets (in our case we only have one)
TRAIN_DS="[/debug/utils/pubmedqa/pubmedqa_train.jsonl]"
VALID_DS="[/debug/utils/pubmedqa/pubmedqa_val.jsonl]"
TEST_DS="[/debug/utils/pubmedqa/pubmedqa_test.jsonl]"

# These are the names of the test datasets
TEST_NAMES="[pubmedqa]"

# This is the PEFT scheme that we will be using. Set to "ptuning" for P-Tuning instead of LoRA
PEFT_SCHEME="lora"

# This is the concat sampling probability. This depends on the number of files being passed in the train set
# and the sampling probability for each file. In our case, we have one training file. Note sum of concat sampling
# probabilities should be 1.0. For example, with two entries in TRAIN_DS, CONCAT_SAMPLING_PROBS might be
# "[0.3,0.7]". For three entries, CONCAT_SAMPLING_PROBS might be "[0.3,0.1,0.6]"
# NOTE: Your entry must contain a value greater than 0.0 for each file
CONCAT_SAMPLING_PROBS="[1.0]"

# This is the tensor parallel size (splitting tensors among GPUs horizontally)
# See above matrix for proper value for the given model size
TP_SIZE=1

# This is the pipeline parallel size (splitting layers among GPUs vertically)
# See above matrix for proper value for the given model size
PP_SIZE=1

# The number of nodes to run this on
# See above matrix for proper value for the given model size
NODE_COUNT=1

# The number of total GPUs used
GPU_COUNT=1

# Where to store the finetuned model and training artifacts
OUTPUT_DIR="./results"


# Run the PEFT command by appropriately setting the values for the parameters such as the number of steps,
# model checkpoint path, batch sizes etc. For a full reference of parameter
# settings refer to the config at https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/tuning/conf/megatron_gpt_finetuning_config.yaml
#PYTHONPATH=/home/akoumparouli/work/NeMo/:/home/akoumparouli/work/Megatron-LM/:$PYTHONPATH \
python /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py \
    trainer.log_every_n_steps=1 \
    trainer.precision=bf16 \
    trainer.devices=${GPU_COUNT} \
    trainer.num_nodes=1 \
    trainer.val_check_interval=20 \
    trainer.max_steps=50 \
    model.restore_from_path=${MODEL} \
    model.peft.peft_scheme=${PEFT_SCHEME} \
    model.peft.lora_tuning.target_modules=[attention_qkv] \
    model.micro_batch_size=1 \
    model.global_batch_size=1 \
    model.tensor_model_parallel_size=${TP_SIZE} \
    model.pipeline_model_parallel_size=${PP_SIZE} \
    model.megatron_amp_O2=True \
    model.activations_checkpoint_granularity=selective \
    model.activations_checkpoint_num_layers=null \
    model.activations_checkpoint_method=uniform \
    model.optim.name=fused_adam \
    model.optim.lr=1e-4 \
    model.answer_only_loss=True \
    model.data.train_ds.file_names=${TRAIN_DS} \
    model.data.validation_ds.file_names=${VALID_DS} \
    model.data.test_ds.file_names=${TEST_DS} \
    model.data.train_ds.concat_sampling_probabilities=${CONCAT_SAMPLING_PROBS} \
    model.data.train_ds.max_seq_length=2048 \
    model.data.validation_ds.max_seq_length=2048 \
    model.data.train_ds.micro_batch_size=1 \
    model.data.train_ds.global_batch_size=128 \
    model.data.validation_ds.micro_batch_size=1 \
    model.data.validation_ds.global_batch_size=128 \
    model.data.train_ds.num_workers=0 \
    model.data.validation_ds.num_workers=0 \
    model.data.test_ds.num_workers=0 \
    model.data.validation_ds.metric.name=loss \
    model.data.test_ds.metric.name=loss \
    exp_manager.create_wandb_logger=False \
    exp_manager.checkpoint_callback_params.mode=min \
    exp_manager.explicit_log_dir=${OUTPUT_DIR} \
    exp_manager.resume_if_exists=True \
    exp_manager.resume_ignore_no_checkpoint=True \
    exp_manager.create_checkpoint_callback=True \
    exp_manager.checkpoint_callback_params.monitor=validation_loss \
    ++exp_manager.checkpoint_callback_params.save_best_model=False \
    exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
    model.save_nemo_on_validation_end=False
