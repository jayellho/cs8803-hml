import os
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from torch.profiler import profile, schedule, tensorboard_trace_handler, ProfilerActivity
import argparse

import accelerate
from accelerate import Accelerator
from accelerate.utils import MegatronLMPlugin
from model import load_model

from pathlib import Path

# Temporarily suppress stdout for Megatron-LM initialization
class SuppressPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class DummyDataset(Dataset):
    def __init__(self, sequence_length, dataset_size=100):
        self.sequence_length = sequence_length
        self.dataset_size = dataset_size
        
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = torch.ones(self.sequence_length)
        m = torch.ones(self.sequence_length)
        y = torch.ones(self.sequence_length)
        return {'input_ids': x, 'attention_mask': m, 'labels': y}

def train(model_name, total_batch_size, log_dir, sequence_length=2048, device="cuda"):
    # Create the plugin with extra Megatron arguments.
    parent = Path(__file__).resolve().parents[0]
    megatron_plugin = MegatronLMPlugin(other_megatron_args={
        # use relative path
        "vocab_file": str(parent / "vocab.json"),
        "merge_file": str(parent / "merges.txt"),
        "train_iters": 1,
    })

    # Initialize accelerator and wait for all processes.
    with SuppressPrint():
        accelerator = Accelerator(megatron_lm_plugin=megatron_plugin)
    accelerator.wait_for_everyone()
    
    # Load model and tokenizer.
    model = load_model(model_name, device=device)

    # Compute per-device batch size matching train.py style.
    tp = accelerator.state.megatron_lm_plugin.tp_degree
    pp = accelerator.state.megatron_lm_plugin.pp_degree
    per_device_batch_size = total_batch_size // (accelerator.num_processes // (tp * pp))

    # Prepare dummy dataset.
    train_dataset = DummyDataset(sequence_length)
    train_dataloader = DataLoader(train_dataset, batch_size=per_device_batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    
    warmup = 5
    tracing_schedule = schedule(wait=0, warmup=warmup, active=1)
    trace_handler = tensorboard_trace_handler(dir_name=log_dir)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=tracing_schedule,
        on_trace_ready=trace_handler,
        profile_memory=True,
        record_shapes=True,
        with_stack=True
    ) as prof:
        model.train()
        for i, batch in enumerate(train_dataloader):
            with torch.profiler.record_function("Single Iteration"):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            prof.step()
            if i == warmup:
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--total_batch_size", type=int)
    parser.add_argument("--logdir", type=str)
    args = parser.parse_args()
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.system(f"rm -rf {args.logdir}")

    try:
        train(model_name=args.model_name, total_batch_size=args.total_batch_size, log_dir=args.logdir)
    except Exception as e:
        print("----------------------------")
        print("----- Profiling failed -----")
        print("----------------------------")
        print("Error:", e)
    else:
        print("---------------------------------------")
        print("----- Profiling done successfully -----")
        print("---------------------------------------")
    finally:
        # Clean up distributed resources if initialized.
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
