"""
Utility functions for NeMo reasoning model inference.

This module provides clean, educational functions for:
1. Exporting NeMo checkpoints to HuggingFace format
2. Loading and applying LoRA adapters 
3. Running inference with streaming support
4. Comparing base vs fine-tuned model responses

Designed for teaching purposes - keeps complex logic here while exposing
simple APIs in the Jupyter notebook.
"""

import os
import torch
import warnings
from pathlib import Path
from typing import Iterator, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

warnings.filterwarnings("ignore")


def find_latest_checkpoint(search_dir: Path) -> Optional[Path]:
    """
    Find the most recent checkpoint directory containing .ckpt files.
    
    Args:
        search_dir: Directory to search for checkpoints
        
    Returns:
        Path to the latest checkpoint directory, or None if not found
    """
    checkpoint_dirs = []
    
    for root, dirs, files in os.walk(search_dir):
        # Look for directories containing .ckpt files
        if any(f.endswith('.ckpt') for f in files):
            checkpoint_dirs.append(Path(root))
    
    if not checkpoint_dirs:
        return None
    
    # Return the most recently modified checkpoint
    return max(checkpoint_dirs, key=lambda p: p.stat().st_mtime)


def export_checkpoint_to_hf(checkpoint_search_dir: str, output_path: str) -> bool:
    """
    Export a NeMo checkpoint to HuggingFace PEFT format.
    
    Args:
        checkpoint_search_dir: Directory to search for checkpoints
        output_path: Where to save the exported adapter
        
    Returns:
        True if export successful, False otherwise
    """
    try:
        # Import here to avoid issues if not in NeMo environment
        from nemo.collections.llm.api import export_ckpt
        from reasoning_sft_utils import find_latest_checkpoint as find_ckpt_alt
        
        # Try to find the latest checkpoint
        checkpoint_path = find_latest_checkpoint(Path(checkpoint_search_dir))
        
        if checkpoint_path is None:
            # Fallback to the existing utility
            try:
                checkpoint_path = find_ckpt_alt(Path(checkpoint_search_dir))
            except ImportError:
                print("❌ Could not find checkpoint and utilities not available")
                return False
        
        if checkpoint_path is None:
            print(f"❌ No checkpoints found in {checkpoint_search_dir}")
            return False
        
        print(f"📥 Found checkpoint: {checkpoint_path}")
        print(f"📤 Exporting to: {output_path}")
        
        # Export using NeMo API
        result = export_ckpt(
            path=str(checkpoint_path),
            target="hf-peft",           # Export LoRA adapter format
            output_path=output_path,    # Where to save HF format
            overwrite=True             # Overwrite existing files
        )
        
        print(f"✅ Export completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return False


def format_chat_prompt(prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
    """
    Format a user prompt using LLaMA 3.1 chat template.
    
    Args:
        prompt: User's question or prompt
        system_prompt: System message to control model behavior
        
    Returns:
        Properly formatted chat prompt
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    return messages


def generate_response(
    model,
    tokenizer,
    prompt: str,
    system_prompt: str = "detailed thinking on",
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    do_sample: bool = True
) -> str:
    """
    Generate a response from the model.
    
    Args:
        model: The loaded model (can be base model or PEFT model)
        tokenizer: Associated tokenizer
        prompt: User's question or prompt
        system_prompt: System message to control behavior
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling
        
    Returns:
        Generated response text
    """
    # Format the prompt with chat template
    messages = format_chat_prompt(prompt, system_prompt)
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Tokenize the input
    inputs = tokenizer(
        formatted_prompt, 
        return_tensors="pt", 
        padding=True, 
        add_special_tokens=False
    )
    
    # Move to model device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens (excluding input prompt)
    input_length = inputs['input_ids'].shape[1]
    response_tokens = outputs[0][input_length:]
    response = tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    return response.strip()


def generate_streaming_response(
    model,
    tokenizer,
    prompt: str,
    system_prompt: str = "detailed thinking on",
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    do_sample: bool = True
) -> Iterator[str]:
    """
    Generate a streaming response from the model, yielding tokens as they're generated.
    
    Args:
        model: The loaded model
        tokenizer: Associated tokenizer  
        prompt: User's question or prompt
        system_prompt: System message to control behavior
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling
        
    Yields:
        Individual tokens as they're generated
    """
    # Format the prompt
    messages = format_chat_prompt(prompt, system_prompt)
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Tokenize input
    inputs = tokenizer(
        formatted_prompt, 
        return_tensors="pt", 
        padding=True, 
        add_special_tokens=False
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    input_length = inputs['input_ids'].shape[1]
    
    # Generate tokens one by one
    with torch.no_grad():
        past_key_values = None
        current_input_ids = inputs['input_ids']
        
        for _ in range(max_new_tokens):
            # Get model outputs
            outputs = model(
                input_ids=current_input_ids,
                attention_mask=inputs['attention_mask'] if past_key_values is None else None,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            # Get logits for the last token
            logits = outputs.logits[:, -1, :]
            
            # Apply temperature and sample
            if do_sample and temperature > 0:
                logits = logits / temperature
                probabilities = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probabilities, num_samples=1)
            else:
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Check for EOS token
            if next_token_id.item() == tokenizer.eos_token_id:
                break
            
            # Decode and yield the new token
            new_token = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            yield new_token
            
            # Update for next iteration
            current_input_ids = next_token_id
            past_key_values = outputs.past_key_values
            
            # Update attention mask
            inputs['attention_mask'] = torch.cat([
                inputs['attention_mask'], 
                torch.ones((1, 1), device=model.device)
            ], dim=1)


def compare_models(
    base_model,
    fine_tuned_model, 
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.1
) -> Tuple[str, str]:
    """
    Compare responses from base model vs fine-tuned model.
    
    Args:
        base_model: The base model without adapters
        fine_tuned_model: The model with LoRA adapters applied
        tokenizer: Associated tokenizer
        prompt: Question to ask both models
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Tuple of (base_response, fine_tuned_response)
    """
    print(f"🔍 Comparing models on: {prompt}")
    print("=" * 80)
    
    # Get base model response with standard system prompt
    print("🔵 BASE MODEL (Standard Assistant):")
    print("-" * 40)
    base_response = generate_response(
        model=base_model,
        tokenizer=tokenizer,
        prompt=prompt,
        system_prompt="You are a helpful assistant.",
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )
    print(f"{base_response}")
    
    print("\n🟢 FINE-TUNED MODEL (Reasoning Enabled):")
    print("-" * 40)
    ft_response = generate_response(
        model=fine_tuned_model,
        tokenizer=tokenizer,
        prompt=prompt,
        system_prompt="detailed thinking on",
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )
    print(f"{ft_response}")
    
    return base_response, ft_response


def set_random_seed(seed: int = 42):
    """Set random seed for reproducible results."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Set default seed for reproducibility
set_random_seed(42) 

def get_gpu_inference_allocation(base_model):
    # 🔍 Show Actual Device Map Created by "auto"
    print("📋 Actual Device Mapping Created by device_map='auto':")
    print("=" * 60)
    
    # Check if the model has the device map attribute
    if hasattr(base_model, 'hf_device_map'):
        device_map = base_model.hf_device_map
        print(f"🎯 Found device map with {len(device_map)} entries:")
        print()
        
        # Group by device for cleaner display
        device_groups = {}
        for module_name, device in device_map.items():
            device_str = str(device)
            if device_str not in device_groups:
                device_groups[device_str] = []
            device_groups[device_str].append(module_name)
        
        # Display the mapping
        for device, modules in device_groups.items():
            print(f"🖥️ Device {device}:")
            print(f"   📊 {len(modules)} modules assigned")
            print("   📝 Modules:")
            for module in modules[:5]:  # Show first 5 modules
                print(f"      • {module}")
            if len(modules) > 5:
                print(f"      • ... and {len(modules) - 5} more")
            print()
            
        print("💾 Memory Distribution:")
        for device in device_groups.keys():
            allocated = torch.cuda.memory_allocated(f"cuda:{device}") / 1024**3
            print(f"   {device}: {allocated:.2f} GB allocated")
        
    else:
        print("❌ No device map found. Model might be on single device.")
        print(f"   Model device: {next(base_model.parameters()).device}")
        
    print("\n" + "=" * 60)