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

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import wrap, enable_wrap
from typing import Optional, Dict, Tuple, List, Union, Unpack, Sequence, Any
from itertools import chain
from datasets import Dataset, load_dataset
from lightning import Trainer, LightningDataModule, LightningModule
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from transformers import AutoTokenizer, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from flash_attn.bert_padding import unpad_input
from copy import deepcopy

from model.modeling_sophie0 import Sophie0ForCausalLM
from model.configuration_sophie0 import Sophie0Config
from model.utils import CosineLRSchedule

torch.set_float32_matmul_precision("medium")

HF_CACHE = os.environ["HF_HOME"]
# HF_CACHE = "/root/autodl-tmp/hf_cache"
# os.environ["HF_HOME"] = HF_CACHE

class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer: AutoTokenizer, model_config: PretrainedConfig, train_config: argparse.Namespace, **kwargs):
        super().__init__(**kwargs)

        self.tokenizer = tokenizer
        self.model_config = model_config
        self.train_config = train_config

        data_files = glob.glob(self.train_config.data_path + "/**/*.parquet", recursive=True)
        print(f"sft data size: {sum([os.path.getsize(_) / (1024 * 1024) for _ in data_files]):.4f} MB\n")

        datas: Dataset = load_dataset("parquet", data_files=data_files, split="train", streaming=False, trust_remote_code=True, columns=["conversations"], cache_dir=HF_CACHE, num_proc=32)
        datas = datas.shuffle(seed=self.train_config.seed)

        # pre-chunk
        self.datas = datas.map(
            self._preprocess,
            batched=True,
            batch_size=5000,
            num_proc=8,
            remove_columns=datas.column_names,
            cache_file_name=os.path.join(HF_CACHE, "parquet/pretrain/sft_map.cache")
        )
    
    def _preprocess(self, raw):
        texts = []
        for conversation in raw["conversations"]:
            cache = ""
            for chat in conversation:
                if chat["from"] == "human":
                    cache += f"<s><user>{chat["value"]}</s>\n"
                elif chat["from"] == "gpt":
                    cache += f"<s><bot>{chat["value"]}</s>\n"
            
            if cache != "": texts.append(cache[:-1])
        
        outputs = self.tokenizer(texts, add_special_tokens=False)['input_ids']
        
        # packing to max_seqlen
        texts.clear()
        cache = []
        batch_length = 0
        for conversation in outputs:
            if batch_length + len(conversation) <= self.train_config.max_seqlen:
                cache.append(conversation)
                batch_length += len(conversation)
            else:
                texts.append(cache)
                cache.clear()
                batch_length = 0
        
        if len(cache) > 0:
            texts.append(cache)
            cache.clear()
            batch_length = 0

        return {"input_ids": texts}
    
    def process(self, indices: list[int]):
        data = self.datas.select(indices)["input_ids"]
        data = list(chain(*data))

        # generate varlen inputs
        input_ids, labels = data, deepcopy(data)
        cu_seqlens, max_seqlen = [0], 0
        for i in range(len(input_ids)):
            # replace labels with <pad> except for the bot reply
            is_bot_reply = False
            for j in range(len(labels[i])):
                if labels[i][j] == self.model_config.bot_token_id: is_bot_reply = True
                elif labels[i][j] == self.model_config.eos_token_id:
                    if not is_bot_reply: labels[i][j] = self.model_config.pad_token_id
                    is_bot_reply = False
                elif not is_bot_reply: labels[i][j] = self.model_config.pad_token_id

            input_ids[i] = input_ids[i][:-1]
            labels[i] = labels[i][1:]
            max_seqlen = max(max_seqlen, len(input_ids[i]))
            cu_seqlens.append(cu_seqlens[-1] + len(input_ids[i]))
        
        input_ids = torch.tensor(list(chain(*input_ids)), dtype=torch.int64)
        labels = torch.tensor(list(chain(*labels)), dtype=torch.int64)
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)

        return dict(
            input_ids=input_ids,
            labels=labels,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen
        )

    def __getitem__(self, idx: int):
        return idx
    
    def __len__(self):
        return len(self.datas)

class SFTDataModule(LightningDataModule):
    def __init__(self, tokenizer: AutoTokenizer, model_config: PretrainedConfig, train_config: argparse.Namespace, **kwargs):
        super().__init__(**kwargs)

        self.model_config = model_config
        self.train_config = train_config

        self.data = SFTDataset(tokenizer, model_config, train_config, **kwargs)
    
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
class SFTModule(LightningModule):
    def __init__(self, tokenizer: AutoTokenizer, model: PreTrainedModel, model_config: PretrainedConfig, train_config: argparse.Namespace, **kwargs):
        super().__init__(**kwargs)

        self.tokenizer = tokenizer
        self.model = model
        self.model_config = model_config
        self.train_config = train_config

        self._date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self._train_tokens = 0

        self.save_hyperparameters(train_config)
    
    @rank_zero_only
    def on_fit_start(self):
        print(f"--------------- Settings ---------------")
        for (k, v) in self.hparams.items(): print(f"{k}:\t {v}")
        print(f"------------- Architecture -------------")
        print(self.model)
        print(f"Total Param: {sum([p.numel() for p in self.model.parameters()])}")
        print(f"------------ Start Training ------------")
        self._start = time.perf_counter()
    
    # @rank_zero_only
    def on_fit_end(self):
        if self.trainer.global_rank == 0:
            wall_clock = time.perf_counter() - self._start
            print(f"Total wall-clock time: {wall_clock:.4f}")
            print("------------ End Training ------------")
            self.tokenizer.save_pretrained(os.path.join(self.train_config.ckpt_path, self._date))

        if self.trainer.num_devices > 1:
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
                state_dict = self.model.state_dict()
        else:
            state_dict = self.model.state_dict()

        if self.trainer.global_rank == 0:
            torch.save(state_dict, os.path.join(self.train_config.ckpt_path, self._date, "pytorch_model.bin"))

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
    
    def configure_model(self):
        self.model = wrap(self.model, device_id=self.trainer.strategy.root_device)
    
    # from https://github.com/Lightning-AI/pytorch-lightning/issues/13339
    # to solve the vanilla gradient_clip_norm not support FSDP
    def configure_gradient_clipping(
            self,
            optimizer,
            gradient_clip_val: Optional[Union[int, float]] = None,
            gradient_clip_algorithm: Optional[str] = None,
    ):
        assert gradient_clip_algorithm in ('norm', None), gradient_clip_algorithm
        if self.trainer.num_devices > 1:
            self.model.clip_grad_norm_(gradient_clip_val)
        else:
            self.clip_gradients(optimizer, gradient_clip_val, gradient_clip_algorithm)
    
    def forward(self, data: Dict):
        outputs: CausalLMOutputWithPast = self.model(
            input_ids=data["input_ids"],
            labels=data["labels"],
            cu_seqlens=data["cu_seqlens"],
            max_seqlen=data["max_seqlen"],
            return_dict=True
        )
        return outputs
    
    def training_step(self, batch: Dict, batch_idx):
        outputs: CausalLMOutputWithPast = self(batch)
        # varlen
        self._train_tokens += batch["input_ids"].size(0)

        self.log("loss", outputs.loss, prog_bar=True, sync_dist=True)
        self.log("lr", self.optimizers().optimizer.param_groups[0]["lr"], prog_bar=True)
        # self.log("batch_size", batch["input_ids"].size(0), prog_bar=True, sync_dist=True, reduce_fx="sum")
        self.log("train_tokens", self._train_tokens, sync_dist=True, reduce_fx="sum")

        return outputs.loss

def main(train_config: argparse.Namespace):
    pl.seed_everything(train_config.seed)

    _dir = os.path.dirname(os.path.abspath(__file__))
    train_config.data_path = os.path.join(_dir, train_config.data_path)
    train_config.ckpt_path = os.path.join(_dir, train_config.ckpt_path)
    logger_path = train_config.ckpt_path if not os.path.exists("/root/tf-logs") else "/root/tf-logs"

    if not os.path.exists(train_config.ckpt_path): os.makedirs(train_config.ckpt_path)

    model_config = Sophie0Config()
    model_config.num_hidden_layers = 2
    model_config.hidden_size = 512
    model_config.intermediate_size = 2048
    model_config.num_heads = 8

    trainer = Trainer(
        precision=train_config.precision,
        strategy="fsdp" if torch.cuda.device_count() > 1 else "auto",
        max_epochs=train_config.max_epochs if train_config.max_steps == -1 else None,
        max_steps=train_config.max_steps if train_config.max_steps != -1 else -1,
        default_root_dir=train_config.ckpt_path,
        accumulate_grad_batches=train_config.accumulate_grad_batches,
        gradient_clip_val=1.0,
        logger=TensorBoardLogger(logger_path),
        callbacks=ModelCheckpoint(
            every_n_train_steps=train_config.save_steps,
            save_weights_only=True,
            save_on_train_epoch_end=True
        )
    )
    if train_config.max_seqlen > 0: train_config.batch_size = train_config.max_token_per_batch // train_config.max_seqlen

    raw_batch_size = train_config.batch_size
    if trainer.num_devices > 1:
        train_config.batch_size = train_config.batch_size // trainer.num_devices
        train_config.max_token_per_batch = train_config.max_token_per_batch // trainer.num_devices
    if trainer.accumulate_grad_batches > 1: 
        train_config.batch_size = train_config.batch_size // trainer.accumulate_grad_batches
        train_config.max_token_per_batch = train_config.max_token_per_batch // trainer.accumulate_grad_batches

    model = Sophie0ForCausalLM(model_config)
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(os.path.join(_dir, "model", "tokenizer"), use_fast=True, trust_remote_code=True, local_files_only=True)
    datamodule = SFTDataModule(tokenizer, model_config, train_config)

    if train_config.max_steps == -1: train_config.max_steps = (len(datamodule.data) + raw_batch_size - 1) // raw_batch_size

    if train_config.pretrained_ckpt_path != "": model.load_state_dict(torch.load(train_config.pretrained_ckpt_path))
    plmodel = SFTModule(tokenizer, model, model_config, train_config)
    trainer.fit(plmodel, datamodule=datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset Args:
    sft_parser = parser.add_argument_group("sft")
    sft_parser.add_argument("--seed", type=int, default=17)
    sft_parser.add_argument("--data_path", type=str, help="Path to the dataset dir", default="data/sft")
    sft_parser.add_argument("--ckpt_path", type=str, help="Path to the checkpoint", default="result/sft")
    sft_parser.add_argument("--pretrained_ckpt_path", type=str, default="", help="Path to the pretrained checkpoint")

    sft_parser.add_argument("--max_seqlen", type=int, default=2048)
    sft_parser.add_argument("--max_token_per_batch", type=int, default=524288)
    sft_parser.add_argument("--batch_size", type=int, default=128)
    sft_parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients for every n batches")
    sft_parser.add_argument("--num_workers", type=int, default=4)

    sft_parser.add_argument("--max_steps", type=int, default=-1)
    sft_parser.add_argument("--max_epochs", type=int, default=1)
    sft_parser.add_argument("--save_steps", type=int, default=5000)
    sft_parser.add_argument("--max_lr", type=float, default=1e-4, help="Maximum learning rate")
    sft_parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
    sft_parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Ratio of steps to warm up learning rate")
    sft_parser.add_argument("--precision", type=str, default="bf16-mixed")

    args = parser.parse_args()

    args.max_token_per_batch = 16384
    args.save_steps = 50
    args.accumulate_grad_batches = 4

    main(args)