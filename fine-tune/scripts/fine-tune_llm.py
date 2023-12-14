# import required libraries
import json
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, HfArgumentParser, pipeline, logging
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
import bitsandbytes as bnb
import huggingface_hub


def find_all_linear_names(model):
    """Function to get list of target modules for parameter efficient fine-tuning"""
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if 'lm_head' in lora_module_names:
            lora_module_names.remove('lm_head')
    return list(lora_module_names)


def fine_tune(config_path):
    """Primary function to fine-tune LLMs"""
    with open(config_path) as config_file:
        config = json.load(config_file)

    # Load dataset (you can process it here)
    dataset = load_dataset(config["dataset_name"], split="train")
    dataset = dataset.shuffle(seed=1234)

    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, config["bnb_4bit_compute_dtype"])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config["use_4bit"],
        bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config["use_nested_quant"],
    )

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and config["use_4bit"]:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=bnb_config,
        device_map=config["device_map"]
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Enable gradient checkpointing and prep for quantized training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # Get target modules for PEFT
    target_modules = find_all_linear_names(model)
    print(target_modules)

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True, add_eos_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=target_modules,
        r=config["lora_r"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Get peft model
    model = get_peft_model(model, peft_config)

    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        optim=config["optim"],
        save_steps=config["save_steps"],
        logging_steps=config["logging_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        fp16=config["fp16"],
        bf16=config["bf16"],
        max_grad_norm=config["max_grad_norm"],
        max_steps=config["max_steps"],
        warmup_ratio=config["warmup_ratio"],
        group_by_length=config["group_by_length"],
        lr_scheduler_type=config["lr_scheduler_type"],
        # report_to="tensorboard"
    )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
        tokenizer=tokenizer,
        args=training_arguments,
        packing=config["packing"],
    )

    # Train model
    trainer.train()

    # Save trained model
    trainer.model.save_pretrained(config["new_model"])

    # Reload base model
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=config["device_map"],
    )

    # Merge base model with adapters and save tokenizer
    merged_model = PeftModel.from_pretrained(base_model, config["new_model"])
    merged_model = merged_model.merge_and_unload()
    merged_model.save_pretrained("merged_model", safe_serialization=True)
    tokenizer.save_pretrained("merged_model")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return merged_model, tokenizer, config


def push_to_hf(merged_model, tokenizer, config):
    """Function to upload merged model to HuggingFace"""
    # Login to HuggingFace hub
    huggingface_hub.login(os.env.get("hf_token"))

    # Push the model and tokenizer to the Hugging Face Model Hub
    merged_model.push_to_hub(config["new_model"], use_temp_dir=False)
    tokenizer.push_to_hub(config["new_model"], use_temp_dir=False)


def main():
    """Driver function"""
    merged_model, tokenizer, config = fine_tune("config_mistral_7b.json") # config_llama_7b.json
    push_to_hf(merged_model, tokenizer, config)


if __name__=="__main__":
    main()
