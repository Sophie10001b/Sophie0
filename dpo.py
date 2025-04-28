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
from copy import deepcopy
from torch_scatter import scatter_mean

from model.modeling_sophie0 import Sophie0ForCausalLM
from model.configuration_sophie0 import Sophie0Config
from model.utils import CosineLRSchedule

torch.set_float32_matmul_precision("medium")

# HF_CACHE = os.environ["HF_HOME"]
HF_CACHE = "/root/autodl-tmp/hf_cache"
os.environ["HF_HOME"] = HF_CACHE

class DPODataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer: AutoTokenizer, model_config: PretrainedConfig, train_config: argparse.Namespace, **kwargs):
        super().__init__(**kwargs)

        self.tokenizer = tokenizer
        self.model_config = model_config
        self.train_config = train_config

        data_files = glob.glob(self.train_config.data_path + "/**/*.parquet", recursive=True)
        print(f"dpo data size: {sum([os.path.getsize(_) / (1024 * 1024) for _ in data_files]):.4f} MB\n")

        datas: Dataset = load_dataset("parquet", data_files=data_files, split="train", streaming=False, trust_remote_code=True, columns=["prompt", "chosen", "rejected"], cache_dir=HF_CACHE, num_proc=32)
        datas = datas.shuffle(seed=self.train_config.seed)

        # pre-chunk
        self.datas = datas.map(
            self._preprocess,
            batched=True,
            batch_size=5000,
            num_proc=32,
            remove_columns=datas.column_names,
            cache_file_name=os.path.join(HF_CACHE, "parquet/dpo/dpo_map.cache")
        )
    
    def _preprocess(self, raw):
        texts = []
        
        pair_cache = []
        for conversations in [raw["chosen"], raw["rejected"]]:
            texts = []
            for conversation in conversations:
                cache = ""
                # only select one-turn conversation
                if len(conversation) < 2: continue
                if conversation[0]['role'] != 'user' or conversation[1]['role'] != 'assistant': continue

                cache = f"<s><user>{conversation[0]['content']}</s>\n" + f"<s><bot>{conversation[1]['content']}</s>"
                texts.append(cache)
            
            outputs = self.tokenizer(texts, add_special_tokens=False)['input_ids']
            pair_cache.append(outputs)
        
        texts.clear()
        
        chosen_batch_length, rejected_batch_length = 0, 0
        chosen_cache, rejected_cache = [], []
        for chosen, rejected in zip(pair_cache[0], pair_cache[1]):
            if max(chosen_batch_length + len(chosen), rejected_batch_length + len(rejected)) <= self.train_config.max_token_per_batch:
                chosen_cache.append(chosen)
                rejected_cache.append(rejected)
                chosen_batch_length += len(chosen)
                rejected_batch_length += len(rejected)
            elif len(chosen_cache) > 0 and len(rejected_cache) > 0:
                texts.append([chosen_cache, rejected_cache])
                chosen_cache, rejected_cache = [], []
                chosen_batch_length, rejected_batch_length = 0, 0
        
        if len(chosen_cache) > 0 and len(rejected_cache) > 0:
            texts.append([chosen_cache, rejected_cache])
            chosen_cache, rejected_cache = [], []
            chosen_batch_length, rejected_batch_length = 0, 0
        
        return {"input_ids": texts}
    
    def process(self, indices: list[int]):
        chosen, rejected = self.datas.select(indices)["input_ids"][0]

        # generate varlen inputs
        results = {"chosen": None, "rejected": None}
        for split, data in zip(["chosen", "rejected"], [chosen, rejected]):
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

            results[split] = dict(
                input_ids=input_ids,
                labels=labels,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen
            )

        return results

    def __getitem__(self, idx: int):
        return idx
    
    def __len__(self):
        return len(self.datas)

class DPODataModule(LightningDataModule):
    def __init__(self, tokenizer: AutoTokenizer, model_config: PretrainedConfig, train_config: argparse.Namespace, **kwargs):
        super().__init__(**kwargs)

        self.model_config = model_config
        self.train_config = train_config

        self.data = DPODataset(tokenizer, model_config, train_config, **kwargs)
    
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
class DPOModule(LightningModule):
    def __init__(self, tokenizer: AutoTokenizer, model: PreTrainedModel, model_config: PretrainedConfig, train_config: argparse.Namespace, **kwargs):
        super().__init__(**kwargs)

        self.tokenizer = tokenizer
        self.model = model
        self.model_config = model_config
        self.train_config = train_config

        # copy a reference model
        self.ref_model = deepcopy(self.model)

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
        self.ref_model = wrap(self.ref_model, device_id=self.trainer.strategy.root_device)

        self.ref_model.eval()
        self.ref_model.requires_grad_(False)
    
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
    
    def _compute_logprob(self, hidden_state: torch.FloatTensor, input_datas: Dict):
        # varlen calculate, with Shape (B * L, D_vocab)
        logits = hidden_state.log_softmax(dim=-1)
        labels = input_datas["labels"]
        cu_seqlens = input_datas["cu_seqlens"]
        actual_logits = torch.gather(logits, dim=-1, index=input_datas["labels"].unsqueeze(-1))[..., 0]
        actual_logits[labels == self.model_config.pad_token_id] = 0

        # scatter the valid token
        batch_length = cu_seqlens[1:] - cu_seqlens[:-1]
        scatter_index = torch.arange(1, batch_length.size(0) + 1, device=cu_seqlens.device, dtype=torch.int64)
        scatter_index = scatter_index.repeat_interleave(batch_length)
        scatter_index[labels == self.model_config.pad_token_id] = 0

        # index=0 is the ignore tokens
        actual_logits = scatter_mean(actual_logits, scatter_index, dim=-1, dim_size=batch_length.size(0) + 1)
        return actual_logits[1:]

    def forward(self, data: Dict):
        # DPO forward
        chosen, rejected = data["chosen"], data["rejected"]

        # ref model output
        ref_chosen: CausalLMOutputWithPast = self.ref_model(
            input_ids=chosen["input_ids"],
            labels=chosen["labels"],
            cu_seqlens=chosen["cu_seqlens"],
            max_seqlen=chosen["max_seqlen"],
            return_dict=True,
            use_cache=False
        )
        ref_rejected: CausalLMOutputWithPast = self.ref_model(
            input_ids=rejected["input_ids"],
            labels=rejected["labels"],
            cu_seqlens=rejected["cu_seqlens"],
            max_seqlen=rejected["max_seqlen"],
            return_dict=True,
            use_cache=False
        )

        ref_chosen_logits, ref_rejected_logits = self._compute_logprob(ref_chosen.logits, chosen), self._compute_logprob(ref_rejected.logits, rejected)
        
        policy_chosen: CausalLMOutputWithPast = self.model(
            input_ids=chosen["input_ids"],
            labels=chosen["labels"],
            cu_seqlens=chosen["cu_seqlens"],
            max_seqlen=chosen["max_seqlen"],
            return_dict=True
        )

        policy_rejected: CausalLMOutputWithPast = self.model(
            input_ids=rejected["input_ids"],
            labels=rejected["labels"],
            cu_seqlens=rejected["cu_seqlens"],
            max_seqlen=rejected["max_seqlen"],
            return_dict=True
        )
        
        policy_chosen_logits, policy_rejected_logits = self._compute_logprob(policy_chosen.logits, chosen), self._compute_logprob(policy_rejected.logits, rejected)

        # compute DPO loss
        total_logits = (policy_chosen_logits - ref_chosen_logits) - (policy_rejected_logits - ref_rejected_logits)
        total_loss = -torch.nn.functional.logsigmoid(self.train_config.beta * total_logits)
        chosen_reward = (policy_chosen_logits - ref_chosen_logits).clone().detach()
        rejected_reward = (policy_rejected_logits - ref_rejected_logits).clone().detach()

        outputs = dict(
            loss=total_loss.mean(),
            chosen_reward=chosen_reward.mean(),
            rejected_reward=rejected_reward.mean(),
        )
        return outputs
    
    def training_step(self, batch: Dict, batch_idx):
        outputs: Dict = self(batch)

        self.log("loss", outputs["loss"], prog_bar=True, sync_dist=True)
        self.log("chosen_reward", outputs["chosen_reward"], prog_bar=False, sync_dist=True)
        self.log("rejected_reward", outputs["rejected_reward"], prog_bar=False, sync_dist=True)
        self.log("lr", self.optimizers().optimizer.param_groups[0]["lr"], prog_bar=True)
        self.log("steps", self.trainer.global_step, prog_bar=True, logger=False)

        return outputs["loss"]

def main(train_config: argparse.Namespace):
    pl.seed_everything(train_config.seed)

    _dir = os.path.dirname(os.path.abspath(__file__))
    train_config.data_path = os.path.join(_dir, train_config.data_path)
    train_config.ckpt_path = os.path.join(_dir, train_config.ckpt_path)
    logger_path = train_config.ckpt_path if not os.path.exists("/root/tf-logs") else "/root/tf-logs"

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
        logger=TensorBoardLogger(logger_path, name="dpo"),
        callbacks=ModelCheckpoint(
            dirpath=train_config.ckpt_path,
            every_n_epochs=1 if train_config.save_steps == -1 else None,
            every_n_train_steps=train_config.save_steps if train_config.save_steps != -1 else None,
            save_weights_only=True,
            save_on_train_epoch_end=True
        )
    )

    assert train_config.batch_size == 1 and train_config.max_token_per_batch > 1
    raw_batch_size = train_config.batch_size * trainer.num_devices * trainer.accumulate_grad_batches
    if trainer.num_devices > 1:
        train_config.max_token_per_batch = train_config.max_token_per_batch // trainer.num_devices
    if trainer.accumulate_grad_batches > 1: 
        train_config.max_token_per_batch = train_config.max_token_per_batch // trainer.accumulate_grad_batches

    model = Sophie0ForCausalLM(model_config)
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(os.path.join(_dir, "model", "tokenizer"), use_fast=True, trust_remote_code=True, local_files_only=True)
    datamodule = DPODataModule(tokenizer, model_config, train_config)

    if train_config.max_steps == -1: train_config.max_steps = ((len(datamodule.data) + raw_batch_size - 1) // raw_batch_size) * train_config.max_epochs

    if train_config.pretrained_ckpt_path != "": model.load_state_dict(torch.load(train_config.pretrained_ckpt_path, map_location='cpu', weights_only=True))
    plmodel = DPOModule(tokenizer, model, model_config, train_config)
    trainer.fit(plmodel, datamodule=datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset Args:
    dpo_parser = parser.add_argument_group("dpo")
    dpo_parser.add_argument("--seed", type=int, default=17)
    dpo_parser.add_argument("--data_path", type=str, help="Path to the dataset dir", default="data/dpo")
    dpo_parser.add_argument("--ckpt_path", type=str, help="Path to the checkpoint", default="result/dpo")
    dpo_parser.add_argument("--pretrained_ckpt_path", type=str, default="", help="Path to the pretrained checkpoint")

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

    main(args)