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

from typing import Optional, Dict, Tuple, List, Union, Unpack, Sequence, Any
from itertools import chain
from datasets import Dataset, load_dataset
from lightning import Trainer, LightningDataModule, LightningModule
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.callbacks import ModelCheckpoint
from transformers import AutoTokenizer, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from flash_attn.bert_padding import unpad_input

from model.modeling_sophie0 import Sophie0ForCausalLM
from model.configuration_sophie0 import Sophie0Config
from model.utils import CosineLRSchedule

class PretrainDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer: AutoTokenizer, model_config: PretrainedConfig, train_config: argparse.Namespace, **kwargs):
        super().__init__(**kwargs)

        self.tokenizer = tokenizer
        self.model_config = model_config
        self.train_config = train_config

        data_files = glob.glob(self.train_config.data_path + "/**/*.parquet", recursive=True)
        
        # static results about the dataset
        domain_stats = {}
        language_stats = {}
        for _data_dir in data_files:
            _dir = _data_dir[len(self.train_config.data_path)+1:]
            domain = _dir.split("/")[0]
            language = _dir.split("/")[1]
            quality = _dir.split("/")[2]
            data_size = os.path.getsize(_data_dir) / (1024 * 1024)

            if domain not in domain_stats: domain_stats[domain] = data_size
            else: domain_stats[domain] = domain_stats[domain] + data_size

            if language not in language_stats: language_stats[language] = data_size
            else: language_stats[language] = language_stats[language] + data_size
        
        print("Language:")
        for (k, v) in language_stats.items(): print(f"{k}:\t {v:.2f} MB")
        print("\nDomain:")
        for (k, v) in domain_stats.items(): print(f"{k}:\t {v:.2f} MB")
        print(f"\nTotal:\t {sum(domain_stats.values()):.2f} MB\n")

        datas: Dataset = load_dataset("parquet", data_files=data_files, split="train", streaming=False, trust_remote_code=True, columns=["text"])
        datas = datas.shuffle(seed=kwargs.pop("seed", 17))

        # pre-chunk
        self.datas = datas.map(
            self._preprocess,
            batched=True,
            batch_size=5000,
            num_proc=os.cpu_count(),
            remove_columns=datas.column_names
        )
    
    def _preprocess(self, raw):
        pad_token_id = self.model_config.pad_token_id
        texts = [self.tokenizer.bos_token + _ + self.tokenizer.eos_token for _ in raw['text']]
        outputs = self.tokenizer(texts, add_special_tokens=False)['input_ids']
        
        texts = list(chain(*outputs))
        texts = [texts[i:i+self.train_config.max_seqlen+1] if i + self.train_config.max_seqlen + 1 <= len(texts) else texts[i:i+self.train_config.max_seqlen+1] + [pad_token_id for j in range(i + self.train_config.max_seqlen + 1 - len(texts))] for i in range(0, len(texts), self.train_config.max_seqlen+1)]

        return {"input_ids": texts}
    
    def process(self, indices: list[int]):
        data = self.datas.select(indices)["input_ids"]
        data = torch.tensor(data, dtype=torch.int64)

        while data.numel() > self.train_config.max_token_per_batch: data = data[:-1]

        return dict(
            input_ids=data[:, :-1],
            labels=data[:, 1:],
        )

    def __getitem__(self, idx: int):
        return idx
    
    def __len__(self):
        return len(self.datas)

class PretrainDataModule(LightningDataModule):
    def __init__(self, tokenizer: AutoTokenizer, model_config: PretrainedConfig, train_config: argparse.Namespace, **kwargs):
        super().__init__(**kwargs)

        self.model_config = model_config
        self.train_config = train_config

        self.data = PretrainDataset(tokenizer, model_config, train_config, **kwargs)
    
    def setup(self, stage):
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.data,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            num_workers=self.train_config.num_workers,
            collate_fn=self.data.process,
            pin_memory=True
        )

#########################################################
#                  --- model ---
#########################################################
class PretrainModule(LightningModule):
    def __init__(self, tokenizer: AutoTokenizer, model: PreTrainedModel, model_config: PretrainedConfig, train_config: argparse.Namespace, **kwargs):
        super().__init__(**kwargs)

        self.tokenizer = tokenizer
        self.model = model
        self.model_config = model_config
        self.train_config = train_config

        self.save_hyperparameters(train_config)
    
    @rank_zero_only
    def on_fit_start(self):
        print(f"--------------- Settings ---------------")
        for (k, v) in self.hparams.items(): print(f"{k}:\t {v}")
        print(f"------------- Architecture -------------")
        print(self.model)
        print(f"Total Param: {sum([p.numel() for p in self.model.parameters()])}")
        print(f"------------ Start Training ------------")
        self._date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self._start = time.perf_counter()
    
    @rank_zero_only
    def on_fit_end(self):
        self.tokenizer.save_pretrained(os.path.join(self.train_config.ckpt_path, self._date))
        self.model.save_pretrained(os.path.join(self.train_config.ckpt_path, self._date))

        wall_clock = time.perf_counter() - self._start
        print(f"Total wall-clock time: {wall_clock:.4f}")
        print("------------ End Training ------------")
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            self.hparams.max_lr,
            eps=self.model_config.eps,
            weight_decay=0.1,
            betas=(0.9, 0.98)
        )
        schedule = CosineLRSchedule(
            optimizer,
            warmup=int(self.hparams.warmup_ratio * self.hparams.max_steps),
            max_lr=self.hparams.max_lr,
            min_lr=self.hparams.min_lr,
            max_steps=self.hparams.max_steps
        )
        return [optimizer], [{"scheduler": schedule, "interval": "step", "frequency": 1}]
    
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()
    
    def forward(self, data: Dict):
        outputs: CausalLMOutputWithPast = self.model(
            input_ids=data["input_ids"],
            labels=data["labels"],
            return_dict=True
        )
        return outputs
    
    def training_step(self, batch: Dict, batch_idx):
        outputs: CausalLMOutputWithPast = self(batch)

        record_dict = {}
        record_dict.update(
            loss=outputs.loss,
            lr=self.optimizers().optimizer.param_groups[0]["lr"]
        )
        self.log_dict(record_dict, prog_bar=True, batch_size=batch["input_ids"].size(0))

        return outputs.loss

def main(train_config: argparse.Namespace):
    _dir = os.path.dirname(os.path.abspath(__file__))
    train_config.data_path = os.path.join(_dir, train_config.data_path)
    train_config.ckpt_path = os.path.join(_dir, train_config.ckpt_path)

    if not os.path.exists(train_config.ckpt_path): os.makedirs(train_config.ckpt_path)

    model_config = Sophie0Config()
    model_config.num_hidden_layers = 2
    model_config.hidden_size = 512
    model_config.intermediate_size = 2048
    model_config.num_heads = 8

    trainer = Trainer(
        precision=train_config.precision,
        strategy=FSDPStrategy() if torch.cuda.device_count() > 1 else "auto",
        max_epochs=train_config.max_epochs,
        default_root_dir=train_config.ckpt_path,
        accumulate_grad_batches=train_config.accumulate_grad_batches,
        gradient_clip_val=1.0,
        callbacks=ModelCheckpoint(every_n_train_steps=train_config.save_steps)
    )
    if train_config.max_seqlen > 0: train_config.batch_size = train_config.max_token_per_batch // train_config.max_seqlen
    if trainer.num_devices > 1:
        args.batch_size = args.batch_size // trainer.num_devices
        args.max_token_per_batch = args.max_token_per_batch // trainer.num_devices
    if trainer.accumulate_grad_batches > 1: 
        args.batch_size = args.batch_size // trainer.accumulate_grad_batches
        args.max_token_per_batch = args.max_token_per_batch // trainer.accumulate_grad_batches

    model = Sophie0ForCausalLM(model_config)
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(os.path.join(_dir, "model", "tokenizer"), use_fast=True, trust_remote_code=True, local_files_only=True)
    data_module = PretrainDataModule(tokenizer, model_config, train_config)

    train_config.max_steps = (len(data_module.data) + train_config.batch_size - 1) // train_config.batch_size

    plmodel = PretrainModule(tokenizer, model, model_config, train_config)
    trainer.fit(plmodel, datamodule=data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset Args:
    pretrain_parser = parser.add_argument_group("paths")
    pretrain_parser.add_argument("--seed", type=int, default=17)
    pretrain_parser.add_argument("--data_path", type=str, help="Path to the dataset dir", default="data/pretrain")
    pretrain_parser.add_argument("--ckpt_path", type=str, help="Path to the checkpoint", default="result/pretrain")

    pretrain_parser.add_argument("--max_seqlen", type=int, default=2048)
    pretrain_parser.add_argument("--max_token_per_batch", type=int, default=1048576)
    pretrain_parser.add_argument("--batch_size", type=int, default=128)
    pretrain_parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients for every n batches")
    pretrain_parser.add_argument("--num_workers", type=int, default=4)

    pretrain_parser.add_argument("--max_steps", type=int, default=100000)
    pretrain_parser.add_argument("--max_epochs", type=int, default=1)
    pretrain_parser.add_argument("--save_steps", type=int, default=5000)
    pretrain_parser.add_argument("--max_lr", type=float, default=1e-4, help="Maximum learning rate")
    pretrain_parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
    pretrain_parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Ratio of steps to warm up learning rate")
    pretrain_parser.add_argument("--precision", type=str, default="bf16-mixed")

    args = parser.parse_args()

    args.max_token_per_batch = 16384
    args.save_steps = 50
    args.accumulate_grad_batches = 4

    main(args)