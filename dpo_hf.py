import os
import time
import glob
import argparse
import pandas
import json
import numpy as np
import torch
import torch.nn as nn
import transformers
import lightning as pl

from copy import deepcopy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import wrap
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer
from typing import Optional, Dict, Tuple, List, Union, Unpack, Sequence, Any
from itertools import chain
from datasets import Dataset, load_dataset
from copy import deepcopy

from model.modeling_sophie0 import Sophie0ForCausalLM
from model.configuration_sophie0 import Sophie0Config

torch.set_float32_matmul_precision("medium")

HF_CACHE = os.environ["HF_HOME"]
# HF_CACHE = "/root/autodl-tmp/hf_cache"
# os.environ["HF_HOME"] = HF_CACHE

def hf_data_process(raw, train_config: argparse.Namespace):
    prompt, chosen, rejected = [], [], []
    for i in range(len(raw["prompt"])):
        if raw["chosen"][i][1]["role"] == "assistant" and raw["rejected"][i][1]["role"] == "assistant":
            if len(raw["prompt"][i]) + max(len(raw["chosen"][i][1]["content"]), len(raw["rejected"][i][1]["content"])) > train_config.max_token_per_batch: continue
            prompt.append(
                f"<s><user>{raw["prompt"][i]}</s>\n<s><bot>"
            )
            chosen.append(
                f"{raw["chosen"][i][1]["content"]}</s>\n"
            )
            rejected.append(
                f"{raw["rejected"][i][1]["content"]}</s>\n"
            )
    
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }

def main(train_config: argparse.Namespace):
    pl.seed_everything(train_config.seed)

    _dir = os.path.dirname(os.path.abspath(__file__))
    train_config.data_path = os.path.join(_dir, train_config.data_path)
    train_config.ckpt_path = os.path.join(_dir, train_config.ckpt_path)
    logger_path = train_config.ckpt_path if not os.path.exists("/root/tf-logs") else "/root/tf-logs"

    if not os.path.exists(train_config.ckpt_path): os.makedirs(train_config.ckpt_path)

    model_config = Sophie0Config()
    model_config.right_shift = True
    model_config.num_hidden_layers = 1
    model_config.hidden_size = 512
    model_config.intermediate_size = 2048

    assert train_config.batch_size == 1 and train_config.max_token_per_batch > 1

    model = Sophie0ForCausalLM(model_config)
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(os.path.join(_dir, "model", "tokenizer"), use_fast=True, trust_remote_code=True, local_files_only=True)
    tokenizer.chat_template = ""

    if train_config.pretrained_ckpt_path != "": model.load_state_dict(torch.load(train_config.pretrained_ckpt_path, map_location='cpu', weights_only=True))

    args.max_token_per_batch = args.max_token_per_batch // (torch.cuda.device_count() * args.accumulate_grad_batches)

    # load dataset
    data_files = glob.glob(train_config.data_path + "/**/*.parquet", recursive=True)
    print(f"dpo data size: {sum([os.path.getsize(_) / (1024 * 1024) for _ in data_files]):.4f} MB\n")

    datas: Dataset = load_dataset("parquet", data_files=data_files, split="train", streaming=False, trust_remote_code=True, cache_dir=HF_CACHE, num_proc=4)
    datas = datas.shuffle(seed=train_config.seed)

    datas = datas.map(
        lambda x: hf_data_process(x, train_config),
        batched=True,
        batch_size=5000,
        num_proc=4,
        remove_columns=datas.column_names,
        load_from_cache_file=True,
        cache_file_name=os.path.join(HF_CACHE, "parquet/dpo/dpo_map_hf.cache")
    )

    dpo_training_args = DPOConfig(
        output_dir=args.ckpt_path,
        overwrite_output_dir=True,
        learning_rate=train_config.max_lr,
        warmup_ratio=train_config.warmup_ratio,
        lr_scheduler_type="cosine",
        num_train_epochs=train_config.max_epochs,
        per_device_train_batch_size=train_config.batch_size,
        gradient_accumulation_steps=train_config.accumulate_grad_batches,
        weight_decay=0.1,
        adam_beta2=0.98,
        adam_epsilon=1e-5,
        save_strategy="epoch",
        save_total_limit=1,
        logging_dir=logger_path,
        logging_steps=20,
        save_only_model=True,
        bf16=True
    )
    dpo_trainer = DPOTrainer(
        model=model,
        train_dataset=datas,
        args=dpo_training_args,
        processing_class=tokenizer
    )

    dpo_trainer.train()
    dpo_trainer.save_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset Args:
    dpo_parser = parser.add_argument_group("dpo")
    dpo_parser.add_argument("--seed", type=int, default=17)
    dpo_parser.add_argument("--data_path", type=str, help="Path to the dataset dir", default="data/dpo")
    dpo_parser.add_argument("--ckpt_path", type=str, help="Path to the checkpoint", default="result/dpo")
    dpo_parser.add_argument("--pretrained_ckpt_path", type=str, default="", help="Path to the pretrained checkpoint")

    dpo_parser.add_argument("--num_devices", type=int, default=1)
    dpo_parser.add_argument("--max_seqlen", type=int, default=2048)
    dpo_parser.add_argument("--max_token_per_batch", type=int, default=524288)
    dpo_parser.add_argument("--batch_size", type=int, default=1)
    dpo_parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients for every n batches")
    dpo_parser.add_argument("--num_workers", type=int, default=4)

    dpo_parser.add_argument("--max_steps", type=int, default=-1)
    dpo_parser.add_argument("--max_epochs", type=int, default=1)
    dpo_parser.add_argument("--save_steps", type=int, default=-1)
    dpo_parser.add_argument("--max_lr", type=float, default=1e-4, help="Maximum learning rate")
    dpo_parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
    dpo_parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Ratio of steps to warm up learning rate")
    dpo_parser.add_argument("--precision", type=str, default="bf16-mixed")

    dpo_parser.add_argument("--beta", type=float, default=0.1, help="Beta for DPO")

    args = parser.parse_args()
    args.max_token_per_batch = 16384

    main(args)