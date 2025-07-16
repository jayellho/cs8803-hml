import torch
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def load_model(model_name, device="cuda"):
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(config)
    return model.to(device)

def input_provider(micro_batch_size, sequence_length, device="cuda"):
    return torch.ones(micro_batch_size, sequence_length, dtype=torch.int32, device=device)
