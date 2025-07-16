import sys
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, default_data_collator
from accelerate import Accelerator
from accelerate.utils import MegatronLMPlugin
from model import load_model
import argparse
import os
from pathlib import Path

# Temporarily suppress stdout for Megatron-LM initialization
class SuppressPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def train(model_name, total_batch_size, block_size=2048, max_train_steps=2):
    parent = Path(__file__).resolve().parents[0]
    megatron_plugin = MegatronLMPlugin(other_megatron_args={
        # use relative path
        "vocab_file": str(parent / "vocab.json"),
        "merge_file": str(parent / "merges.txt"),
        "train_iters": 2,
    })

    # Initialize accelerator and wait for all processes.
    with SuppressPrint():
        accelerator = Accelerator(megatron_lm_plugin=megatron_plugin)
    accelerator.wait_for_everyone()
    
    # Download and load dataset.
    dataset_name = "wikitext"
    dataset_config_name = "wikitext-2-raw-v1"
    raw_datasets = load_dataset(dataset_name, dataset_config_name)
    
    # Load model and tokenizer.
    model = load_model(model_name, device="cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    total_params = sum(param.numel() for param in model.parameters())


    # Tokenize texts.
    column_names = raw_datasets["train"].column_names
    text_column = "text" if "text" in column_names else column_names[0]
    def tokenize_fn(examples):
        return tokenizer(examples[text_column])
    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_fn,
            batched=True,
            remove_columns=column_names,
            desc="Tokenizing dataset",
        )

    # Adjust block_size if needed.
    if block_size > tokenizer.model_max_length:
        block_size = min(block_size, tokenizer.model_max_length)

    # Group tokens into blocks.
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(concatenated.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                  for k, t in concatenated.items()}
        result["labels"] = result["input_ids"].copy()
        return result

    with accelerator.main_process_first():
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            desc=f"Grouping texts into chunks of {block_size}",
        )

    # Prepare a subset of the training data.
    train_dataset = lm_datasets["train"].select(range(10))

    # Compute per device batch size (defaulting TP and PP to 1 if not available).
    tp = accelerator.state.megatron_lm_plugin.tp_degree
    pp = accelerator.state.megatron_lm_plugin.pp_degree
    per_device_batch_size = total_batch_size // (accelerator.num_processes // (tp * pp))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator,
        batch_size=per_device_batch_size
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    
    # Training loop.
    print("***** Running training *****")
    print(f"Instantaneous batch size per device = {per_device_batch_size}")
    print(f"Total train batch size = {total_batch_size}")
    print(f"Total optimization steps = {max_train_steps}")

    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            completed_steps += 1
        if completed_steps >= max_train_steps:
            break

    return total_params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--total_batch_size", type=int)
    args = parser.parse_args()
    total_params = train(model_name=args.model_name, total_batch_size=args.total_batch_size)
    return total_params

if __name__ == "__main__":
    try:
        total_params = main()
    except Exception as e:
        # print only on the main process.
        if (torch.distributed.get_rank() == 0):
            print("-------------------------------")
            print("----- Training run failed -----")
            print("-------------------------------")
            print("Error:", e)
    else:
        if (torch.distributed.get_rank() == 0):
            print("------------------------------------------")
            print("----- Training run done successfully -----")
            print("------------------------------------------")
            print("")
            print("Parameter counts of the trained model : {:.2f} Billion".format(total_params / 1e9))
            print("")

    finally:
        # Clean up distributed resources if initialized.
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
