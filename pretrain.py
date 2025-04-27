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

from model.modeling_sophie0 import Sophie0ForCausalLM
from model.configuration_sophie0 import Sophie0Config
from model.utils import CosineLRSchedule

torch.set_float32_matmul_precision("medium")

HF_CACHE = "/root/autodl-tmp/hf_cache"
os.environ["HF_HOME"] = HF_CACHE

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

        datas: Dataset = load_dataset("parquet", data_files=data_files, split="train", streaming=False, trust_remote_code=True, columns=["text"], cache_dir=HF_CACHE, num_proc=32)
        datas = datas.shuffle(seed=self.train_config.seed)

        # pre-chunk
        self.datas = datas.map(
            self._preprocess,
            batched=True,
            batch_size=5000,
            num_proc=32,
            remove_columns=datas.column_names,
            cache_file_name=os.path.join(HF_CACHE, "parquet/pretrain/pretrain_map.cache")
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

        input_ids = data[:, :-1]
        labels = data[:, 1:]

        while input_ids.numel() > self.train_config.max_token_per_batch:
            input_ids = input_ids[:-1]
            labels = labels[:-1]

        return dict(
            input_ids=input_ids,
            labels=labels
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
            print("finish saving model")

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
            return_dict=True
        )
        return outputs
    
    def training_step(self, batch: Dict, batch_idx):
        outputs: CausalLMOutputWithPast = self(batch)
        self._train_tokens += batch["input_ids"].size(0) * batch["input_ids"].size(1)

        self.log("loss", outputs.loss, prog_bar=True, sync_dist=True)
        self.log("lr", self.optimizers().optimizer.param_groups[0]["lr"], prog_bar=True)
        self.log("batch_size", batch["input_ids"].size(0), prog_bar=True, sync_dist=True, reduce_fx="sum")
        self.log("train_tokens", self._train_tokens, sync_dist=True, reduce_fx="sum")

        return outputs.loss

def main(train_config: argparse.Namespace):
    pl.seed_everything(train_config.seed)

    _dir = os.path.dirname(os.path.abspath(__file__))
    train_config.data_path = os.path.join(_dir, train_config.data_path)
    train_config.ckpt_path = os.path.join(_dir, train_config.ckpt_path)

    if not os.path.exists(train_config.ckpt_path): os.makedirs(train_config.ckpt_path)

    model_config = Sophie0Config()

    trainer = Trainer(
        precision=train_config.precision,
        strategy="fsdp" if torch.cuda.device_count() > 1 else "auto",
        max_epochs=train_config.max_epochs if train_config.max_steps == -1 else None,
        max_steps=train_config.max_steps if train_config.max_steps != -1 else -1,
        default_root_dir=train_config.ckpt_path,
        accumulate_grad_batches=train_config.accumulate_grad_batches,
        gradient_clip_val=1.0,
        logger=TensorBoardLogger("/root/tf-logs", name="pretrain"),
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
    datamodule = PretrainDataModule(tokenizer, model_config, train_config)

    if train_config.max_steps == -1: train_config.max_steps = (len(datamodule.data) + raw_batch_size - 1) // raw_batch_size

    plmodel = PretrainModule(tokenizer, model, model_config, train_config)
    trainer.fit(plmodel, datamodule=datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset Args:
    pretrain_parser = parser.add_argument_group("pretrain")
    pretrain_parser.add_argument("--seed", type=int, default=17)
    pretrain_parser.add_argument("--data_path", type=str, help="Path to the dataset dir", default="data/pretrain")
    pretrain_parser.add_argument("--ckpt_path", type=str, help="Path to the checkpoint", default="result/pretrain")

    pretrain_parser.add_argument("--max_seqlen", type=int, default=2048)
    pretrain_parser.add_argument("--max_token_per_batch", type=int, default=524288)
    pretrain_parser.add_argument("--batch_size", type=int, default=128)
    pretrain_parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients for every n batches")
    pretrain_parser.add_argument("--num_workers", type=int, default=4)

    pretrain_parser.add_argument("--max_steps", type=int, default=-1)
    pretrain_parser.add_argument("--max_epochs", type=int, default=1)
    pretrain_parser.add_argument("--save_steps", type=int, default=5000)
    pretrain_parser.add_argument("--max_lr", type=float, default=1e-4, help="Maximum learning rate")
    pretrain_parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
    pretrain_parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Ratio of steps to warm up learning rate")
    pretrain_parser.add_argument("--precision", type=str, default="bf16-mixed")

    args = parser.parse_args()

    main(args)