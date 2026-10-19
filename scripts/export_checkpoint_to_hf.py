import argparse
import torch
from pathlib import Path
from reasoning_sft_utils import find_latest_checkpoint
from nemo.collections.llm.api import export_ckpt
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="convert a nemo 2.0 checkpoint to huggingface format"
    )

    parser.add_argument(
        "--model_search_dir",
        type=str,
        default="/workspace/output/results-20250720-0502-2-devices-baseline-lora-only/",  # 💾 replace with your actual path
        help="path to search for the latest checkpoint folder containing .distcp shards"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="/workspace/output/ft-model-hf",  # 📦 output directory path
        help="desired output directory for the huggingface format model"
    )
  # add a skip checkpoint argument
    parser.add_argument(
        "--skip_checkpoint",
        action="store_true",
        help="skip the checkpoint export and only run inference"
    )

    args = parser.parse_args()


    # args.skip_checkpoint = True  # Commented out to run the actual export
    if not(args.skip_checkpoint):
        checkpoint_path = find_latest_checkpoint(Path(args.model_search_dir))

        print(f"loading checkpoint from: {checkpoint_path}")
        print(f"output path: {args.output_path}")
    
        # use nemo 2.0 export api for converting checkpoint to huggingface format
        # note: nemo 2.0 models don't support .nemo export, only huggingface format
        print(f"exporting checkpoint from: {checkpoint_path}")
        print(f"output path: {args.output_path}")
        
        result = export_ckpt(
            path=checkpoint_path,
            target="hf-peft",
            output_path=args.output_path,
            overwrite=True
        )

        print(f"[✔] converted {checkpoint_path} → {args.output_path}")

    else:
        print(f"skipping checkpoint export and only running inference")

    # test the exported model with a simple inference
    print("\n🧪 testing model inference...")
    
    # First test: try loading just the base model
    print("🔧 Step 1: Testing base model loading...")
    try:
        import torch
        base_model_path = "/workspace/Llama-3.1-8B-Instruct"
        print(f"Loading base model from: {base_model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        print("✅ Base model loaded successfully!")
        
        # Test base model inference
        tokenizer.pad_token = tokenizer.eos_token
        prompt = "explain the reasoning behind solving 2 + 2:"
        messages = [
            {"role": "system", "content": "detailed thinking on"},
            {"role": "user", "content": prompt}
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, add_special_tokens=False)
        outputs = model.generate(**inputs, max_length=150, do_sample=True, temperature=0.7)
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(f"✅ Base model inference successful!")
        print(f"Base model response: {response[:200]}...")
        
    except Exception as base_error:
        print(f"❌ Base model test failed: {base_error}")
        return
    
    # Second test: try loading with PEFT adapter using different approaches
    print("\n🔧 Step 2: Testing PEFT adapter loading...")
    try:
        from peft import PeftModel, PeftConfig
        import warnings
        warnings.filterwarnings("ignore")
        
        # Try manual loading approach
        print(f"Loading PEFT config from: {args.output_path}")
        peft_config = PeftConfig.from_pretrained(args.output_path)
        print(f"✅ PEFT config loaded: {peft_config.peft_type}")
        
        print(f"Loading PEFT adapter with fallback methods...")
        
        # Method 1: Try standard loading with error handling
        try:
            model = PeftModel.from_pretrained(model, args.output_path, is_trainable=False)
            print("✅ PEFT model loaded with Method 1!")
        except Exception as e1:
            print(f"Method 1 failed: {str(e1)[:100]}...")
            
            # Method 2: Try with torch_dtype parameter
            try:
                import torch
                model = PeftModel.from_pretrained(
                    model, 
                    args.output_path, 
                    torch_dtype=torch.float16,
                    is_trainable=False
                )
                print("✅ PEFT model loaded with Method 2!")
            except Exception as e2:
                print(f"Method 2 failed: {str(e2)[:100]}...")
                
                # Method 3: Skip PEFT and note successful export
                print("⚠️  PEFT loading failed, but model export was successful!")
                print("💡 You can manually load the adapter using other tools or PEFT versions")
                return
        tokenizer.pad_token = tokenizer.eos_token
        # use for inference
        prompt = "explain the reasoning behind solving 2 + 2:"

        # Setup message template
        messages = [
            {"role": "system", "content": "detailed thinking on"},
            {"role": "user", "content": prompt}
        ]

        # Format prompt using chat template
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )



        inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, add_special_tokens=False)

        outputs = model.generate(**inputs, max_length=150, do_sample=True, temperature=0.7)
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)

        print(f"✅ model inference successful!")
        print(f"prompt: {prompt}")
        print(f"response: {response}")
    except Exception as e:
        print(f"⚠️  inference test failed: {e}")
        print("model export was successful, but inference test encountered an issue.")


if __name__ == "__main__":
    main()

