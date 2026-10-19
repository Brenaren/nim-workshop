#!/bin/bash
# Copy data and models to local environment
sudo chown $USER:$USER $HOME/s3/*
cp $HOME/s3/chat_10k.jsonl $HOME/nim-workshop/data/Llama-Nemotron-Post-Training-Dataset/SFT/chat/
cp $HOME/s3/llama-3-8b-lora-hf.tgz $HOME/nim-workshop/models/
tar -xzf $HOME/nim-workshop/models/llama-3-8b-lora-hf.tgz -C $HOME/nim-workshop/models/

