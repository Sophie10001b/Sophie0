import os
import re
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
from transformers import AutoTokenizer, PreTrainedModel, PretrainedConfig, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from flash_attn.bert_padding import unpad_input
from copy import deepcopy
from torchmetrics import MeanMetric, SumMetric, Metric

from model.modeling_sophie0 import Sophie0ForCausalLM
from model.configuration_sophie0 import Sophie0Config
from model.utils import CosineLRSchedule

torch.set_float32_matmul_precision("medium")

# HF_CACHE = os.environ["HF_HOME"]
HF_CACHE = "/root/autodl-tmp/hf_cache"
os.environ["HF_HOME"] = HF_CACHE

class GRPODataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer: AutoTokenizer, model_config: PretrainedConfig, train_config: argparse.Namespace, **kwargs):
        super().__init__(**kwargs)

        self.tokenizer = tokenizer
        self.model_config = model_config
        self.train_config = train_config

        data_files = glob.glob(self.train_config.data_path + "/**/*.parquet", recursive=True)
        print(f"grpo data size: {sum([os.path.getsize(_) / (1024 * 1024) for _ in data_files]):.4f} MB\n")

        datas: Dataset = load_dataset("parquet", data_files=data_files, split="train", streaming=False, trust_remote_code=True, columns=["label", "conversations"], cache_dir=HF_CACHE, num_proc=32)
        datas = datas.shuffle(seed=self.train_config.seed)

        # pre-chunk
        self.datas = datas.map(
            self._preprocess,
            batched=True,
            batch_size=5000,
            num_proc=32,
            remove_columns=datas.column_names,
            cache_file_name=os.path.join(HF_CACHE, "parquet/grpo/grpo_map.cache")
        )
    
    def _preprocess(self, raw):
        prompt = []
        answer = []
        for label, conversation in zip(raw["label"], raw["conversations"]):
            if (conversation[0]["from"] == "user" and conversation[1]["from"] == "gpt_reasoning" and conversation[2]["from"] == "gpt_output"):
                content = f"<s><user>{conversation[0]["value"]}</s>"
                prompt.append(content)

                # generate answer pattern
                cache = []
                for name, solution in zip(label["names"], label["solution"]):
                    character = "knight" if solution else "knave"
                    cache.append(rf"[\s]*\([\d]\)[\s]?{name} is a [kK]{character[1:]}")
                answer.append(tuple(cache))

        return dict(
            input_dis=prompt,
            labels=answer
        )
    
    def process(self, indices: list[int]):
        raw_data = self.datas.select(indices)
        input_ids, labels = raw_data["input_ids"], raw_data["labels"]

        outputs = self.tokenizer(input_ids, return_tensors="pt", padding="longest", padding_side="right")
        
        input_ids = torch.tensor(outputs.input_ids, dtype=torch.int64)
        attention_mask = torch.tensor(outputs.attention_mask, dtype=torch.int64)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask
        )

    def __getitem__(self, idx: int):
        return idx
    
    def __len__(self):
        return len(self.datas)

class GRPODataModule(LightningDataModule):
    def __init__(self, tokenizer: AutoTokenizer, model_config: PretrainedConfig, train_config: argparse.Namespace, **kwargs):
        super().__init__(**kwargs)

        self.model_config = model_config
        self.train_config = train_config

        self.data = GRPODataset(tokenizer, model_config, train_config, **kwargs)
    
    def setup(self, stage):
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.data,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            num_workers=self.train_config.num_workers,
            collate_fn=self.data.process,
            persistent_workers=True,
            pin_memory=True
        )

#########################################################
#                  --- model ---
#########################################################
class GRPOModule(LightningModule):
    def __init__(self, tokenizer: AutoTokenizer, model: PreTrainedModel, ref_model: PreTrainedModel, model_config: PretrainedConfig, train_config: argparse.Namespace, **kwargs):
        super().__init__(**kwargs)

        self.tokenizer = tokenizer
        self.model = model
        self.ref_model = ref_model
        self.model_config = model_config
        self.train_config = train_config

        self._date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self._train_tokens = 0

        self.ref_model.eval()
        self.ref_model.requires_grad_(False)

        self.save_hyperparameters(train_config)

        self.generation_config = GenerationConfig(
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=4096,
            do_sample=True,
            top_k=20,
            top_p=0.7,
            temperature=0.8,
            num_beams=1,
            repeat_penalty=1.1,
            use_cache=True
        )
    
    @rank_zero_only
    def on_fit_start(self):
        print(f"--------------- Settings ---------------")
        for (k, v) in self.hparams.items(): print(f"{k}:\t {v}")
        print(f"------------- Architecture -------------")
        print(self.model)
        print(f"Total Param: {sum([p.numel() for p in self.model.parameters()])}")
        print(f"------------ Start Training ------------")
        self._start = time.perf_counter()

        self.logger.log_hyperparams(self.train_config)
        # metrics
        self.train_metrics = {
            "loss": MeanMetric().to(self.trainer.strategy.root_device),
            "tokens": SumMetric().to(self.trainer.strategy.root_device),
            "train_tokens": SumMetric().to(self.trainer.strategy.root_device),
        }
    
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
        if self.trainer.num_devices > 1 and isinstance(self.trainer.strategy, FSDPStrategy):
            self.model.clip_grad_norm_(gradient_clip_val)
        else:
            self.clip_gradients(optimizer, gradient_clip_val, gradient_clip_algorithm)
    
    def _compute_reward_for_each_rollout(self, rollout: str, answer: List[str]) -> float:
        # 1st: compute format reward
        format_reward = 0.
        if rollout.count("<think>") == 1: format_reward += 1/3
        if rollout.count("</think>") == 1: format_reward += 1/3
        if rollout.count("</s>") == 1: format_reward += 1/3

        # 2nd: compute answer reward
        answer_reward = True
        for _answer in answer:
            if len(re.findall(_answer, rollout)) != 1: answer_reward = False
        
        if answer_reward: answer_reward = 1.
        return self.train_config.answer_scale * answer_reward + self.train_config.format_scale * format_reward

    def _compute_reward(self, rollout: List[str], answer: List[List[str]]):
        rollout_reward = []
        for i in range(0, len(rollout), self.train_config.rollout):
            grouped_rollout = rollout[i:i+self.train_config.rollout]
            grouped_answer = answer[i]

            for _rollout in grouped_rollout:
                _rollout_reward = self._compute_reward_for_each_rollout(_rollout, grouped_answer)


    def forward(self, data: Dict):
        # Step 1: generate response for each request
        self.model.eval()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                rollout = self.model.generate(
                    input_ids=data["input_ids"].repeat_interleave(self.train_config.rollout_num, dim=0),
                    attention_mask=data["attention_mask"].repeat_interleave(self.train_config.rollout_num, dim=0),
                    use_cache=True,
                    use_varlen_inference=True,
                    generation_config=self.generation_config
                )
                rollout = self.tokenizer.batch_decode(rollout, skip_special_tokens=False)


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

        self.train_metrics["loss"].update(outputs.loss)
        self.train_metrics["tokens"].update(batch["input_ids"].size(0))
        self.train_metrics["train_tokens"].update(batch["input_ids"].size(0))

        if batch_idx % self.trainer.accumulate_grad_batches == 0:
            loss = self.train_metrics["loss"].compute()
            lr = self.optimizers().optimizer.param_groups[0]["lr"]
            steps = self.trainer.global_step
            tokens = self.train_metrics["tokens"].compute()

            self.log_dict(
                dict(loss=loss, lr=lr, steps=steps, tokens=tokens),
                prog_bar=True,
                logger=False
            )
            self.log_dict({"sft/loss": loss, "sft/lr": lr})

            self.train_metrics["loss"].reset()
            self.train_metrics["tokens"].reset()

        return outputs.loss

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
        logger=WandbLogger(
            project="Sophie0",
            name="reasoning_grpo",
            save_dir=logger_path
        ),
        callbacks=ModelCheckpoint(
            every_n_epochs=1 if train_config.save_steps == -1 else None,
            every_n_train_steps=train_config.save_steps if train_config.save_steps != -1 else None,
            save_weights_only=True,
            save_on_train_epoch_end=True
        ),
        log_every_n_steps=20
    )

    assert train_config.batch_size == 1 and train_config.max_token_per_batch > 1
    raw_batch_size = train_config.batch_size * trainer.num_devices * trainer.accumulate_grad_batches
    if trainer.num_devices > 1:
        train_config.max_token_per_batch = train_config.max_token_per_batch // trainer.num_devices
    if trainer.accumulate_grad_batches > 1: 
        train_config.max_token_per_batch = train_config.max_token_per_batch // trainer.accumulate_grad_batches

    model = Sophie0ForCausalLM(model_config)
    ref_model = Sophie0ForCausalLM(model_config)
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(os.path.join(_dir, "model", "tokenizer"), use_fast=True, trust_remote_code=True, local_files_only=True)
    datamodule = SFTDataModule(tokenizer, model_config, train_config)

    if train_config.max_steps == -1: train_config.max_steps = ((len(datamodule.data) + raw_batch_size - 1) // raw_batch_size) * train_config.max_epochs

    if train_config.pretrained_ckpt_path != "": 
        _state_dict = torch.load(train_config.pretrained_ckpt_path, map_location='cpu', weights_only=True)
        model.load_state_dict(_state_dict)
        ref_model.load_state_dict(_state_dict)

    plmodel = GRPOModule(tokenizer, model, model_config, train_config)
    trainer.fit(plmodel, datamodule=datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset Args:
    grpo_parser = parser.add_argument_group("grpo")
    grpo_parser.add_argument("--seed", type=int, default=17)
    grpo_parser.add_argument("--data_path", type=str, help="Path to the dataset dir", default="data/reasoning")
    grpo_parser.add_argument("--ckpt_path", type=str, help="Path to the checkpoint", default="result/grpo_reasoning")
    grpo_parser.add_argument("--pretrained_ckpt_path", type=str, default="", help="Path to the pretrained checkpoint")

    grpo_parser.add_argument("--max_seqlen", type=int, default=2048)
    grpo_parser.add_argument("--max_token_per_batch", type=int, default=524288)
    grpo_parser.add_argument("--batch_size", type=int, default=1)
    grpo_parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients for every n batches")
    grpo_parser.add_argument("--num_workers", type=int, default=4)

    grpo_parser.add_argument("--max_steps", type=int, default=-1)
    grpo_parser.add_argument("--max_epochs", type=int, default=1)
    grpo_parser.add_argument("--save_steps", type=int, default=-1)
    grpo_parser.add_argument("--max_lr", type=float, default=1e-4, help="Maximum learning rate")
    grpo_parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
    grpo_parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Ratio of steps to warm up learning rate")
    grpo_parser.add_argument("--precision", type=str, default="bf16-mixed")

    grpo_parser.add_argument("--rollout", type=int, default=10)
    grpo_parser.add_argument("--format_scale", type=float, default=0.1)
    grpo_parser.add_argument("--answer_scale", type=float, default=1.0)

    args = parser.parse_args()

    main(args)