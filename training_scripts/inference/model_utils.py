# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from peft import PeftModel, PeftConfig
from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer

# Function to load the main model for text generation
def load_model(model_name, quantization):
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        load_in_8bit=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    return model


# Function to load the PeftModel for performance optimization
def load_peft_model(model, peft_model):
    peft_model = PeftModel.from_pretrained(model, peft_model)
    return peft_model

def load_peft_model_v2(peft_model_path, quantization):
    # Load peft config for pre-trained checkpoint etc.
    config = PeftConfig.from_pretrained(peft_model_path)

    # load base LLM model and tokenizer
    model = LlamaForCausalLM.from_pretrained(
        config.base_pre_train_model_path,
        return_dict=True,
        load_in_8bit=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    tokenizer = LlamaTokenizer.from_pretrained(config.base_pre_train_model_path)

    # Load the Lora model
    model = PeftModel.from_pretrained(model, peft_model_path)
    model.eval()
    print("Peft model loaded")

    return model, tokenizer


# Loading the model from config to load FSDP checkpoints into that
def load_llama_from_config(config_path):
    model_config = LlamaConfig.from_pretrained(config_path) 
    model = LlamaForCausalLM(config=model_config)
    return model
    

if __name__ == '__main__':
    base_model = load_model('meta-llama/Llama-2-7b-hf', False)
    peft_model = load_peft_model(base_model, peft_model='llama-2-7b-fine-tuning')

    peft_model = load_peft_model_v2('llama-2-7b-fine-tuning', False)