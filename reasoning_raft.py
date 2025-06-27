import os
import time
import glob
import re
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
from lightning.pytorch.strategies import FSDPStrategy, ModelParallelStrategy
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from transformers import AutoTokenizer, PreTrainedModel, PretrainedConfig, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from flash_attn.bert_padding import unpad_input
from copy import deepcopy
from torchmetrics import MeanMetric, SumMetric, Metric, Accuracy
from swanlab.integration.pytorch_lightning import SwanLabLogger

from model.modeling_sophie0 import Sophie0ForCausalLM
from model.configuration_sophie0 import Sophie0Config
from model.utils import CosineLRSchedule
from model.parallelism_sophie0 import parallelize_setting

torch.set_float32_matmul_precision("medium")

# HF_CACHE = os.environ["HF_HOME"]
HF_CACHE = "/root/autodl-tmp/hf_cache"
os.environ["HF_HOME"] = HF_CACHE

class KKDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer: AutoTokenizer, model_config: PretrainedConfig, train_config: argparse.Namespace, **kwargs):
        super().__init__(**kwargs)

        self.tokenizer = tokenizer
        self.model_config = model_config
        self.train_config = train_config

        datas: Dataset = load_dataset("json", data_files=[os.path.join(train_config.data_path, "people3_num100.jsonl")], split="train", streaming=False, trust_remote_code=True, cache_dir=HF_CACHE, num_proc=4)
        self.datas = datas.map(
            self.convert_kk_dataset,
            batched=True,
            batch_size=1000,
            num_proc=4,
            remove_columns=datas.column_names,
            load_from_cache_file=False
        )

    def convert_kk_dataset(self, raw):
        prompts = []
        target = []
        for i, (quiz, names, solution) in enumerate(zip(raw["quiz"], raw["names"], raw["solution"])):
            user_input = f"<s><user>{quiz}</s>\n<s><bot>"
            patterns = []
            for _name, _solution in zip(names, solution):
                character = "knight" if _solution else "knave"
                patterns.append(rf"[\s]*\([\d]\)[\s]?{_name} is a [kK]{character[1:]}")
            
            prompts.append(user_input)
            target.append(patterns)
        
        return {
            "input_ids": prompts,
            "labels": target
        }
    
    def process(self, indices: list[int]):
        input_ids = self.datas.select(indices)["input_ids"]
        labels = self.datas.select(indices)["labels"]
        
        inputs = self.tokenizer(input_ids, return_tensors="pt", padding="longest", padding_side="left")
        return dict(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            labels=labels
        )
    
    def __getitem__(self, idx: int):
        return idx
    
    def __len__(self):
        return len(self.datas)
    

class RAFTDataset(torch.utils.data.Dataset):
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
            num_proc=32,
            remove_columns=datas.column_names,
            cache_file_name=os.path.join(HF_CACHE, "parquet/reasoning/raft_map.cache")
        )
    
    def _preprocess(self, raw):
        tokenized_text = []
        for conversation in raw["conversations"]:
            if (conversation[0]["from"] == "user" and conversation[1]["from"] == "gpt"):
                content = f"<s><user>{conversation[0]["value"]}</s>\n<s><bot>{conversation[1]["value"]}\n"
                content = self.tokenizer(content, add_special_tokens=False)['input_ids']
                if len(content) > self.train_config.max_token_per_batch: continue
                tokenized_text.append(content)

        return {"input_ids": tokenized_text}
    
    def process(self, indices: list[int]):
        data = self.datas.select(indices)["input_ids"]

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

class RAFTDataModule(LightningDataModule):
    def __init__(self, tokenizer: AutoTokenizer, model_config: PretrainedConfig, train_config: argparse.Namespace, **kwargs):
        super().__init__(**kwargs)

        self.model_config = model_config
        self.train_config = train_config

        self.train_data = RAFTDataset(tokenizer, model_config, train_config, **kwargs)
        self.eval_data = KKDataset(tokenizer, model_config, train_config, **kwargs)
    
    def setup(self, stage):
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            num_workers=self.train_config.num_workers,
            collate_fn=self.train_data.process,
            persistent_workers=True,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.eval_data,
            batch_size=self.train_config.eval_batch_size,
            shuffle=False,
            num_workers=self.train_config.num_workers,
            collate_fn=self.eval_data.process,
            persistent_workers=True,
            pin_memory=True
        )

#########################################################
#                  --- model ---
#########################################################
class RAFTModule(LightningModule):
    def __init__(self, tokenizer: AutoTokenizer, model: PreTrainedModel, model_config: PretrainedConfig, train_config: argparse.Namespace, **kwargs):
        super().__init__(**kwargs)

        self.tokenizer = tokenizer
        self.model = model
        self.model_config = model_config
        self.train_config = train_config

        self._date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self._train_tokens = 0

        self.save_hyperparameters(train_config)
    
    def on_fit_start(self):
        if self.global_rank == 0:
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
        self.eval_metrics = {
            "pass@1": Accuracy(task="multiclass", num_classes=2).to(self.trainer.strategy.root_device)
        }
    
    # @rank_zero_only
    def on_fit_end(self):
        if self.trainer.global_rank == 0:
            wall_clock = time.perf_counter() - self._start
            print(f"Total wall-clock time: {wall_clock:.4f}")
            print("------------ End Training ------------")
            self.tokenizer.save_pretrained(os.path.join(self.train_config.ckpt_path, self._date))

        if self.trainer.num_devices > 1 and isinstance(self.trainer.strategy, ModelParallelStrategy):
            sharded_sd = self.model.state_dict()
            state_dict = {}
            for param_name, sharded_param in sharded_sd.items():
                full_param = sharded_param.full_tensor()
                if self.trainer.is_global_zero:
                    state_dict[param_name] = full_param.cpu()
                else:
                    del full_param
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
        if isinstance(self.trainer.strategy, ModelParallelStrategy):
            self.model = parallelize_setting(self.model, self.device_mesh)
    
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
    
    def on_validation_epoch_start(self):
        self.response_cache = []
        for k, v in self.eval_metrics.items():
            if isinstance(v, Metric): v.reset()
    
    def validation_step(self, batch: Dict, batch_idx):
        generate_config = GenerationConfig(
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=2048,
            do_sample=True,
            top_k=20,
            top_p=0.8,
            temperature=0.8,
            num_beams=1,
            repeat_penalty=1.1,
            use_cache=True,
            return_dict_in_generate=True
        )

        self.model.train(False)
        with torch.no_grad():
            rollout = self.model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=True,
                use_varlen_inference=True,
                generation_config=generate_config
            )
            outputs = self.tokenizer.batch_decode(rollout.sequences, skip_special_tokens=False)
            del rollout
            torch.cuda.empty_cache()
        
        self.model.train(True)
        
        correct_cache = [0 for _ in range(len(outputs))]
        for i, output in enumerate(outputs):
            response = output.split("<bot>")[-1]
            if (response.count("<think>") != 1 or response.count("</think>") != 1 or response.count("</s>") != 1): continue
            answer = response.split("</think>")[-1]
            is_correct = True
            answer_patterns = batch["labels"][i]
            for answer_pattern in answer_patterns:
                if re.search(answer_pattern, answer) is None: is_correct = False
            
            if is_correct:
                correct_cache[i] = 1

                if self.trainer.strategy.is_global_zero:
                    prompt, response = output.split("<bot>")
                    prompt = prompt.split("<user>")[-1]
                    response.split("<pad>")[0]
                    self.response_cache.append(dict(
                        prompt=prompt,
                        response=response,
                    ))
        
        correct_cache = torch.tensor(correct_cache, device=self.trainer.strategy.root_device)
        label_cache = torch.ones_like(correct_cache)
        self.eval_metrics["pass@1"].update(correct_cache, label_cache)
    
    def on_validation_epoch_end(self):
        for k, v in self.eval_metrics.items():
            if isinstance(v, Metric):
                self.log(f"eval/{k}", v.compute(), sync_dist=True)
        
        if self.trainer.strategy.is_global_zero and len(self.response_cache) > 0:
            with open(os.path.join(self.train_config.ckpt_path, f"{self.global_step}_eval_correct.txt"), 'w') as f:
                f.write(json.dumps(self.response_cache, indent=4))

def main(train_config: argparse.Namespace):
    pl.seed_everything(train_config.seed)

    _dir = os.path.dirname(os.path.abspath(__file__))
    train_config.data_path = os.path.join(_dir, train_config.data_path)
    train_config.ckpt_path = os.path.join(_dir, train_config.ckpt_path)
    logger_path = train_config.ckpt_path if not os.path.exists("/root/tf-logs") else "/root/tf-logs"

    if not os.path.exists(train_config.ckpt_path): os.makedirs(train_config.ckpt_path)

    model_config = Sophie0Config()
    logger = SwanLabLogger(project="Sophie0", experiment_name="reasoning_raft", logdir=logger_path)
    strategy = ModelParallelStrategy(
        tensor_parallel_size=1,
        data_parallel_size=torch.cuda.device_count(),
        save_distributed_checkpoint=False
    )

    trainer = Trainer(
        precision=train_config.precision,
        strategy=strategy if torch.cuda.device_count() > 1 else "auto",
        max_epochs=train_config.max_epochs if train_config.max_steps == -1 else None,
        max_steps=train_config.max_steps if train_config.max_steps != -1 else -1,
        default_root_dir=train_config.ckpt_path,
        accumulate_grad_batches=train_config.accumulate_grad_batches,
        gradient_clip_val=1.0,
        logger=logger,
        callbacks=ModelCheckpoint(
            every_n_epochs=1 if train_config.save_steps == -1 else None,
            every_n_train_steps=train_config.save_steps if train_config.save_steps != -1 else None,
            save_weights_only=True,
            save_on_train_epoch_end=True
        ),
        log_every_n_steps=10,
        val_check_interval=0.1
    )

    raw_batch_size = train_config.batch_size
    if trainer.num_devices > 1:
        train_config.batch_size = train_config.batch_size // trainer.num_devices
        train_config.eval_batch_size = train_config.eval_batch_size // trainer.num_devices
    if trainer.accumulate_grad_batches > 1: 
        train_config.batch_size = train_config.batch_size // trainer.accumulate_grad_batches

    model = Sophie0ForCausalLM(model_config)
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(os.path.join(_dir, "model", "tokenizer"), use_fast=True, trust_remote_code=True, local_files_only=True)
    datamodule = RAFTDataModule(tokenizer, model_config, train_config)

    if train_config.max_steps == -1: train_config.max_steps = ((len(datamodule.train_data) + raw_batch_size - 1) // raw_batch_size) * train_config.max_epochs

    if train_config.pretrained_ckpt_path != "": model.load_state_dict(torch.load(train_config.pretrained_ckpt_path, map_location='cpu', weights_only=True))
    plmodel = RAFTModule(tokenizer, model, model_config, train_config)
    trainer.fit(plmodel, datamodule=datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset Args:
    raft_parser = parser.add_argument_group("raft")
    raft_parser.add_argument("--seed", type=int, default=17)
    raft_parser.add_argument("--data_path", type=str, help="Path to the dataset dir", default="data/reasoning")
    raft_parser.add_argument("--ckpt_path", type=str, help="Path to the checkpoint", default="result/reasoning")
    raft_parser.add_argument("--pretrained_ckpt_path", type=str, default="", help="Path to the pretrained checkpoint")

    raft_parser.add_argument("--max_seqlen", type=int, default=2048)
    raft_parser.add_argument("--max_token_per_batch", type=int, default=524288)
    raft_parser.add_argument("--batch_size", type=int, default=1)
    raft_parser.add_argument("--eval_batch_size", type=int, default=1)
    raft_parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients for every n batches")
    raft_parser.add_argument("--num_workers", type=int, default=4)

    raft_parser.add_argument("--max_steps", type=int, default=-1)
    raft_parser.add_argument("--max_epochs", type=int, default=1)
    raft_parser.add_argument("--save_steps", type=int, default=-1)
    raft_parser.add_argument("--max_lr", type=float, default=1e-4, help="Maximum learning rate")
    raft_parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
    raft_parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Ratio of steps to warm up learning rate")
    raft_parser.add_argument("--precision", type=str, default="bf16-mixed")

    args = parser.parse_args()

    main(args)