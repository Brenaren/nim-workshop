from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Load your fine-tuned model
print("Loading base model and adapter...")
def set_random_seed(seed=42):
    """Set random seed for reproducible results"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(10)

# Configuration
base_model_path = "/workspace/models/Llama-3.1-8B-Instruct"
adapter_path = "/workspace/models/llama-3-8b-lora-hf/"

print(f"🔧 Loading base model from: {base_model_path}")
# Load base model first
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load tokenizer from base model
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# Set pad token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"🔧 Loading PEFT adapter from: {adapter_path}")
# Load PEFT config and adapter
peft_config = PeftConfig.from_pretrained(adapter_path)
print(f"✅ PEFT config loaded: {peft_config.peft_type}")

# Apply the adapter to the base model
ft_model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
print("✅ PEFT model loaded successfully!")

# Keep reference to both models
base_model = model  # Reference to base model without adapter
print("✅ Both base model and fine-tuned model are ready!")

def is_already_formatted(prompt):
    """Check if prompt is already formatted with chat template"""
    return "<|begin_of_text|>" in prompt and "<|start_header_id|>" in prompt

def format_prompt(prompt, system_prompt="You are a helpful assistant."):
    """Format prompt using chat template if needed"""
    if is_already_formatted(prompt):
        return prompt
    else:
        # Apply chat template with system message and user content
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        return tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

def generate_response(prompt, max_new_tokens=2048, temperature=0.1, do_sample=True, use_base_model=False, verbose=False):
    """Generate a response to a given prompt
    
    Args:
        prompt: Input prompt
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling
        use_base_model: If True, use base model; if False, use fine-tuned model
        verbose: If True, print detailed tokenization and debugging info
    """
    
    # Choose which model to use
    model_to_use = base_model if use_base_model else ft_model
    model_name = "BASE MODEL" if use_base_model else "FINE-TUNED MODEL"
    
    # Choose system prompt based on model type
    if use_base_model:
        system_prompt = "You are a helpful assistant."
    else:
        system_prompt = "detailed thinking on"
    
    print(f"🤖 Using: {model_name}")
    if verbose:
        print(f"📝 System prompt: {repr(system_prompt)}")
    
    # Format the prompt with appropriate system message
    formatted_prompt = format_prompt(prompt, system_prompt=system_prompt)
    
    if verbose:
        print(f"Original prompt: {repr(prompt)}")
        print(f"Formatted prompt: {repr(formatted_prompt)}")
    
    # Tokenize input - don't add special tokens since chat template already includes them
    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, add_special_tokens=False)
    
    if verbose:
        print(f"Tokenized input IDs: {inputs['input_ids'].tolist()}")
        print(f"Tokenized input shape: {inputs['input_ids'].shape}")
        print(f"Attention mask: {inputs['attention_mask'].tolist()}")
        
        # Decode tokens to verify
        decoded_input = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
        print(f"Decoded tokens (with special tokens): {repr(decoded_input)}")
        print("-" * 80)
    
    # Move inputs to same device as model
    inputs = {k: v.to(model_to_use.device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model_to_use.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    if verbose:
        print(f"Generated output IDs: {outputs[0].tolist()}")
        print(f"Generated output shape: {outputs[0].shape}")
    
    # Decode only the new tokens (excluding the input prompt)
    input_length = inputs['input_ids'].shape[1]
    response_tokens = outputs[0][input_length:]
    response = tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    if verbose:
        # Also show full output with special tokens
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(f"Full output (with special tokens): {repr(full_output)}")
    
    return response.strip()

def compare_models(prompt, max_new_tokens=2048, temperature=0.1, verbose=False):
    """Compare responses from both base and fine-tuned models"""
    print(f"\n{'='*120}")
    print(f"COMPARING MODELS FOR PROMPT: {prompt}")
    print(f"{'='*120}")
    
    # Get base model response
    print(f"\n{'🔵 BASE MODEL RESPONSE:':<50}")
    print("-" * 80)
    base_response = generate_response(prompt, max_new_tokens, temperature, use_base_model=True, verbose=verbose)
    print(f"Response: {base_response}")
    
    print(f"\n{'🟢 FINE-TUNED MODEL RESPONSE:':<50}")
    print("-" * 80)
    ft_response = generate_response(prompt, max_new_tokens, temperature, use_base_model=False, verbose=verbose)
    print(f"Response: {ft_response}")
    
    print(f"\n{'='*120}")
    return base_response, ft_response

# Example usage
if __name__ == "__main__":
    print("Model loaded successfully!\n")
    
    # Your list of prompts
    prompts = [
        "How many r's are in the word 'strawberry'?",
        "Which number is bigger? 10.119 or 10.19?",
        "You have three boxes: Box A contains two white balls, Box B contains two black balls, and Box C contains one white and one black ball. You pick a box at random and draw one ball at random—it's white. Without putting it back, you draw the second ball from the same box. What is the probability the second ball is white? Show your reasoning step by step."
    ]
    # answers, 3 10.19, 2/3



    #print("Choose mode:")
    #print("1. Fine-tuned model only")
    #print("2. Base model only") 
    #print("3. Compare both models")
    #print("4. Interactive mode")
    
    try:
        choice = "3" #input("Enter choice (1-4): ").strip()
        
        # Ask about verbose mode
        verbose = False
        
        if verbose:
            print("✅ Verbose mode enabled - will show detailed debugging info")
        else:
            print("📝 Verbose mode disabled - minimal output")
            
    except KeyboardInterrupt:
        print("\nExiting...")
        exit()
    
    if choice == "3":
        # Compare both models
        print(f"\n{'='*120}")
        print("COMPARING BASE MODEL vs FINE-TUNED MODEL")
        print(f"{'='*120}")
        
        for i, prompt in enumerate(prompts):
            print(f"\n\nPROMPT {i+1}/{len(prompts)}: {prompt}")
            compare_models(prompt, max_new_tokens=2048, verbose=verbose)
            
            #if i < len(prompts) - 1:  # Don't pause after last prompt
            #    input("\nPress Enter to continue to next prompt...")
    
    elif choice in ["1", "2"]:
        # Single model mode
        use_base = (choice == "2")
        model_name = "BASE MODEL" if use_base else "FINE-TUNED MODEL"
        
        print(f"\n{'='*100}")
        print(f"PROCESSING WITH {model_name}")
        print(f"{'='*100}")
        
        # Process each prompt
        for i, prompt in enumerate(prompts):
            print(f"\n{'='*100}")
            print(f"PROCESSING PROMPT {i+1}/{len(prompts)}")
            print(f"{'='*100}")
            
            response = generate_response(prompt, max_new_tokens=2048, use_base_model=use_base, verbose=verbose)
            
            print(f"\nGenerated Response: {repr(response)}")
            print(f"\nFinal Response: {response}")
            
            print(f"\n{'='*100}")
            print("END OF PROMPT")
            print(f"{'='*100}\n")
            
            # Add a pause between prompts for readability
            if i < len(prompts) - 1:  # Don't pause after last prompt
                input("Press Enter to continue to next prompt...")
    
    elif choice == "4":
        # Interactive mode
        print("\n🎯 Interactive Mode")
        print("Commands:")
        print("  'base: <prompt>' - Use base model")
        print("  'ft: <prompt>' - Use fine-tuned model") 
        print("  'compare: <prompt>' - Compare both models")
        print("  '<prompt>' - Use fine-tuned model (default)")
        print("  'verbose on/off' - Toggle verbose mode")
        print("  'quit' - Exit")
        
        while True:
            try:
                user_input = input("\nEnter command: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if user_input.lower() in ['verbose on', 'verbose true', 'verbose 1']:
                    verbose = True
                    print("✅ Verbose mode enabled")
                    continue
                elif user_input.lower() in ['verbose off', 'verbose false', 'verbose 0']:
                    verbose = False
                    print("📝 Verbose mode disabled")
                    continue
                
                if user_input.startswith('base:'):
                    prompt = user_input[5:].strip()
                    if prompt:
                        response = generate_response(prompt, max_new_tokens=2048, use_base_model=True, verbose=verbose)
                        print(f"🔵 Base Model: {response}")
                
                elif user_input.startswith('ft:'):
                    prompt = user_input[3:].strip()
                    if prompt:
                        response = generate_response(prompt, max_new_tokens=2048, use_base_model=False, verbose=verbose)
                        print(f"🟢 Fine-tuned Model: {response}")
                
                elif user_input.startswith('compare:'):
                    prompt = user_input[8:].strip()
                    if prompt:
                        compare_models(prompt, max_new_tokens=250, verbose=verbose)
                
                elif user_input:
                    # Default to fine-tuned model
                    response = generate_response(user_input, max_new_tokens=2048, use_base_model=False, verbose=verbose)
                    print(f"🟢 Fine-tuned Model: {response}")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
    
    else:
        print("Invalid choice. Exiting...")
    
    print("\nAll prompts processed!")




# format as simple prompt string (one line for everything)
prompt = "You have three boxes: Box A contains two white balls, Box B contains two black balls, and Box C contains one white and one black ball. You pick a box at random and draw one ball at random—it's white. Without putting it back, you draw the second ball from the same box. What is the probability the second ball is white? Show your reasoning step by step."

# compare_models(prompt, max_new_tokens=2048, verbose=verbose)
generate_response(prompt, max_new_tokens=2048, use_base_model=False, verbose=verbose)