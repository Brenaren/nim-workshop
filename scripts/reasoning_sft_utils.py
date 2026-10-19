#!/usr/bin/env python3
"""
Data utilities for reasoning SFT training.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


def find_latest_checkpoint(checkpoint_search_dir: str = "./output") -> str:
    """Find the latest checkpoint directory with context and weights subdirectories."""
    
    if not os.path.exists(checkpoint_search_dir):
        raise FileNotFoundError(f"Checkpoint search directory does not exist: {checkpoint_search_dir}")
    
    checkpoint_dirs = []
    
    # Walk through all directories recursively
    for root, dirs, files in os.walk(checkpoint_search_dir):
        # Check if current directory has both context and weights subdirectories
        if "context" in dirs and "weights" in dirs:
            mtime = os.path.getmtime(root)
            checkpoint_dirs.append((root, mtime))
    
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No directories with context/weights subdirectories found in {checkpoint_search_dir}")
    
    # Sort by modification time (most recent first)
    checkpoint_dirs.sort(key=lambda x: x[1], reverse=True)
    
    latest_checkpoint = checkpoint_dirs[0][0]
    latest_time = datetime.fromtimestamp(checkpoint_dirs[0][1])
    
    print(f"🔍 Found {len(checkpoint_dirs)} checkpoint directories")
    print(f"✓ Latest checkpoint: {latest_checkpoint}")
    print(f"  Modified: {latest_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return latest_checkpoint


def validate_data_format(data_dir: str, expected_format: str):
    """
    Validate that the data in the directory matches the expected format.
    """
    training_file = os.path.join(data_dir, "training.jsonl")
    
    if not os.path.exists(training_file):
        raise FileNotFoundError(f"Training file not found: {training_file}")
    
    # Read first few lines to check format
    with open(training_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 3:  # Check only first 3 lines
                break
            
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in line {i+1}: {e}")
            
            if expected_format == "chat":
                # Check for role-based format
                if "input" in data and isinstance(data["input"], list):
                    # Check if input contains role/content structure
                    if len(data["input"]) > 0 and "role" in data["input"][0] and "content" in data["input"][0]:
                        continue
                    else:
                        raise ValueError(f"Chat format expected but line {i+1} doesn't contain role/content structure")
                else:
                    raise ValueError(f"Chat format expected but line {i+1} doesn't contain input list")
                    
            elif expected_format == "completion":
                # Check for simple input/output format
                if "input" in data and "output" in data:
                    if isinstance(data["input"], str) and isinstance(data["output"], str):
                        continue
                    else:
                        raise ValueError(f"Completion format expected but line {i+1} input/output are not strings")
                else:
                    raise ValueError(f"Completion format expected but line {i+1} doesn't contain input/output fields")
    
    print(f"✓ Data format validation passed for {expected_format} format")


def convert_chat_to_messages_format(input_file: str, output_file: str):
    """
    Convert chat format data to HuggingFace messages format that NeMo can handle.
    
    Input format: {"input": [{"role": "user", "content": "..."}], "output": "...", "system_prompt": "..."}
    Output format: {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    print(f"Converting chat format data from {input_file} to {output_file}")
    
    converted_count = 0
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            try:
                data = json.loads(line.strip())
                
                # Build messages array
                messages = []
                
                # Add system message if available
                system_content = data.get("system_prompt", "You are a helpful assistant.")
                if system_content and system_content.strip():
                    messages.append({"role": "system", "content": system_content})
                
                # Add user messages from input
                if "input" in data and isinstance(data["input"], list):
                    for message in data["input"]:
                        if message.get("role") and message.get("content"):
                            messages.append({
                                "role": message["role"], 
                                "content": message["content"]
                            })
                
                # Add assistant response
                if "output" in data and data["output"]:
                    messages.append({"role": "assistant", "content": data["output"]})
                
                # Create new format
                new_data = {"messages": messages}
                
                # Preserve other metadata (excluding the fields we've processed)
                for key, value in data.items():
                    if key not in ["input", "output", "system_prompt"]:
                        new_data[key] = value
                
                outfile.write(json.dumps(new_data) + "\n")
                converted_count += 1
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    print(f"✓ Converted {converted_count} samples to messages format")
    return converted_count


def prepare_chat_data_for_nemo(data_dir: str) -> str:
    """
    Prepare chat format data for NeMo by converting to HuggingFace messages format.
    Returns the path to the converted data directory.
    """
    converted_dir = f"{data_dir}_converted"
    os.makedirs(converted_dir, exist_ok=True)
    
    # Convert training data
    training_input = os.path.join(data_dir, "training.jsonl")
    training_output = os.path.join(converted_dir, "training.jsonl")
    
    if os.path.exists(training_input):
        convert_chat_to_messages_format(training_input, training_output)
    
    # Convert validation data if it exists
    validation_input = os.path.join(data_dir, "validation.jsonl")
    validation_output = os.path.join(converted_dir, "validation.jsonl")
    
    if os.path.exists(validation_input):
        convert_chat_to_messages_format(validation_input, validation_output)
    
    # Convert test data if it exists
    test_input = os.path.join(data_dir, "test.jsonl")
    test_output = os.path.join(converted_dir, "test.jsonl")
    
    if os.path.exists(test_input):
        convert_chat_to_messages_format(test_input, test_output)
    
    return converted_dir


def detect_format_from_sample(data: Dict[str, Any]) -> str:
    """
    Detect format from a single data sample.
    
    Logic:
    - If data contains 'role' and 'content' keys anywhere -> chat
    - Otherwise -> completion
    """
    # Convert data to string and check for role/content pattern
    data_str = json.dumps(data).lower()
    
    if '"role"' in data_str and '"content"' in data_str:
        return "chat"
    else:
        return "completion" 