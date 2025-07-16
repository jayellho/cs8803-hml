import torch
import transformers
from pathlib import Path
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch

def load_model():
    device = 'cpu'

    model_path = "/content/gpt3_27_1_layer.json"
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_config(config)
    model = model.to(device)
    return model

def input_provider(micro_batch_size, sequence_length):
    device = 'cpu'
    input_ids = torch.ones(
        micro_batch_size, sequence_length, dtype=torch.int32, device=device)
    return input_ids

