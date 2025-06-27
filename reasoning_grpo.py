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
from lightning.pytorch.strategies import FSDPStrategy, ModelParallelStrategy, DDPStrategy
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from transformers import AutoTokenizer, PreTrainedModel, PretrainedConfig, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from flash_attn.bert_padding import unpad_input
from copy import deepcopy
from torch_scatter import scatter
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
                content = f"<s><user>{conversation[0]["value"]}</s>\n<s><bot>"
                prompt.append(content)

                # generate answer pattern
                cache = []
                for name, solution in zip(label["names"], label["solution"]):
                    character = "knight" if solution else "knave"
                    cache.append(rf"[\s]*\([\d]\)[\s]?{name} is a [kK]{character[1:]}")
                answer.append(tuple(cache))

        return dict(
            input_ids=prompt,
            labels=answer
        )
    
    def process(self, indices: list[int]):
        raw_data = self.datas.select(indices)
        input_ids, labels = raw_data["input_ids"], raw_data["labels"]

        outputs = self.tokenizer(input_ids, return_tensors="pt", padding="longest", padding_side="right")

        return dict(
            input_ids=outputs.input_ids,
            labels=labels,
            attention_mask=outputs.attention_mask
        )

    def __getitem__(self, idx: int):
        return idx
    
    def __len__(self):
        return len(self.datas)

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

        if self.ref_model is not None:
            self.ref_model.eval()
            self.ref_model.requires_grad_(False)
        
        self.infer_model = deepcopy(self.model)
        self.infer_model.eval()
        self.infer_model.requires_grad_(False)

        self.save_hyperparameters(train_config)

        self.generation_config = GenerationConfig(
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=1024,
            do_sample=True,
            top_k=20,
            top_p=0.8,
            temperature=0.8,
            num_beams=1,
            repeat_penalty=1.1,
            use_cache=True,
            return_dict_in_generate=True
        )
    
    # @rank_zero_only
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
            "response_length": MeanMetric().to(self.trainer.strategy.root_device),
            "answer_reward": MeanMetric().to(self.trainer.strategy.root_device),
            "format_reward": MeanMetric().to(self.trainer.strategy.root_device),
            "pass_rate": MeanMetric().to(self.trainer.strategy.root_device),
        }
        if self.ref_model is not None:
            self.train_metrics["kl_loss"] = MeanMetric().to(self.trainer.strategy.root_device)

        self.eval_metrics = {
            "pass@1": Accuracy(task="multiclass", num_classes=2).to(self.trainer.strategy.root_device)
        }

        self.last_output_steps = -1
    
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
            self.infer_model = parallelize_setting(self.infer_model, self.device_mesh)
            if self.ref_model is not None:
                self.ref_model = parallelize_setting(self.ref_model, self.device_mesh)
    
    # from https://github.com/Lightning-AI/pytorch-lightning/issues/13339
    # to solve the vanilla gradient_clip_norm not support FSDP
    # def configure_gradient_clipping(
    #         self,
    #         optimizer,
    #         gradient_clip_val: Optional[Union[int, float]] = None,
    #         gradient_clip_algorithm: Optional[str] = None,
    # ):
    #     assert gradient_clip_algorithm in ('norm', None), gradient_clip_algorithm
    #     if self.trainer.num_devices > 1 and isinstance(self.trainer.strategy, FSDPStrategy):
    #         self.model.clip_grad_norm_(gradient_clip_val)
    #     else:
    #         self.clip_gradients(optimizer, gradient_clip_val, gradient_clip_algorithm)
    
    def _compute_reward_for_each_rollout(self, rollout: str, rollout_ids: list[int], answer: List[str]) -> Dict:
        # 1st: compute format reward
        format_reward = 0.
        if rollout.count("<think>") == 1: format_reward += 1/3
        if rollout.count("</think>") == 1: format_reward += 1/3
        if rollout.split("<bot>")[-1].count("</s>") == 1: format_reward += 1/3

        bot_ids = self.tokenizer.vocab["<bot>"]
        eos_ids = self.tokenizer.vocab["</s>"]
        
        start_pos, end_pos = 0, len(rollout_ids)
        for i in range(len(rollout_ids)):
            if rollout_ids[i] == bot_ids:
                start_pos = i
                break
        
        for i in range(len(rollout_ids)-1, 0, -1):
            if rollout_ids[i] == eos_ids:
                end_pos = i
                break
        
        if end_pos <= start_pos: end_pos = len(rollout_ids)
        response_length = end_pos - start_pos

        # 2nd: compute answer reward
        answer_reward = False
        if rollout.count("</think>") == 1:
            response = rollout.split("</think>")[-1]
            answer_reward = True
            for _answer in answer:
                if len(re.findall(_answer, response)) != 1: answer_reward = False
        
        if answer_reward: answer_reward = 1.
        else: answer_reward = 0.

        return dict(
            total_reward=self.train_config.answer_scale * answer_reward + self.train_config.format_scale * format_reward,
            answer_reward=answer_reward,
            format_reward=format_reward,
            response_length=response_length
        )

    def _compute_reward(self, rollout: List[str], rollout_ids: List[List[int]], answer: List[List[str]]) -> Dict:
        rollout_reward = []
        response_length = []
        pass_rate = 0
        for i in range(0, len(rollout), self.train_config.rollout):
            grouped_rollout = rollout[i:i+self.train_config.rollout]
            grouped_rollout_ids = rollout_ids[i:i+self.train_config.rollout]
            grouped_answer = answer[i]
            group_answer_reward = []
            group_format_reward = []

            rollout_reward.append([])
            response_length.append([])
            for j, _rollout in enumerate(grouped_rollout):
                rewards: Dict = self._compute_reward_for_each_rollout(_rollout, grouped_rollout_ids[j], grouped_answer)
                rollout_reward[-1].append(rewards["total_reward"])
                response_length[-1].append(rewards["response_length"])
                group_answer_reward.append(rewards["answer_reward"])
                group_format_reward.append(rewards["format_reward"])
        
            # length penalty for answer
            min_length = [response_length[-1][k] for k in range(len(response_length[-1])) if group_answer_reward[k] == 1]
            if len(min_length) > 0:
                min_length = min(min_length)
                for k in range(len(rollout_reward[-1])):
                    length_penalty = max(1 - self.train_config.length_penalty, min_length / response_length[-1][k])
                    rollout_reward[-1][k] = rollout_reward[-1][k] * length_penalty if group_answer_reward[k] == 1 else rollout_reward[-1][k]
            
            if sum(group_answer_reward) > 0: pass_rate += 1
            group_answer_reward = np.mean(group_answer_reward).item()
            group_format_reward = np.mean(group_format_reward).item()
            
            _mean_reward = float(np.mean(rollout_reward[-1]))
            _std_reward = float(np.std(rollout_reward[-1]))
            rollout_reward[-1] = list(map(lambda x: (x - _mean_reward) / (_std_reward + 1e-5), rollout_reward[-1]))
        
        return dict(
            reward=torch.tensor(list(chain(*rollout_reward)), dtype=torch.float32, device=self.device),
            response_length=np.mean(list(chain(*response_length))).item(),
            answer_reward=group_answer_reward,
            format_reward=group_format_reward,
            pass_rate=(pass_rate / (len(rollout) // self.train_config.rollout))
        )

    def _generate_forward_inputs(self, data: list[list[int]]):
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
        
        input_ids = torch.tensor(list(chain(*input_ids)), dtype=torch.int64, device=self.device)
        labels = torch.tensor(list(chain(*labels)), dtype=torch.int64, device=self.device)
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=self.device)

        return dict(
            input_ids=input_ids,
            labels=labels,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen
        )
    
    def _compute_logprob(
        self,
        hidden_state: torch.FloatTensor,
        input_datas: Dict
    ):
        # varlen calculate, with Shape (B * L, D_vocab)
        prob = hidden_state.log_softmax(dim=-1)
        labels = input_datas["labels"]
        cu_seqlens = input_datas["cu_seqlens"]
        actual_prob = torch.gather(prob, dim=-1, index=input_datas["labels"].unsqueeze(-1)).squeeze(-1)
        actual_prob[labels == self.model_config.pad_token_id] = 0

        # select the target logits in each batch
        labels = torch.split(labels, (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().tolist())
        actual_prob = torch.split(actual_prob, (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().tolist())
        actual_prob = list(map(lambda x, y: x[y != self.model_config.pad_token_id], actual_prob, labels))

        new_cu_seqlens = torch.tensor([0] + [x.size(0) for x in actual_prob], dtype=torch.int32, device=self.device)
        new_cu_seqlens = torch.cumsum(new_cu_seqlens, dim=0)

        return (torch.cat(actual_prob, dim=0), new_cu_seqlens)

    def forward(self, data: Dict):
        # Step 1: generate response for each request
        with torch.no_grad():
            self.infer_model.load_state_dict(deepcopy(self.model.state_dict()))
            rollout = self.infer_model.generate(
                input_ids=data["input_ids"].repeat_interleave(self.train_config.rollout, dim=0),
                attention_mask=data["attention_mask"].repeat_interleave(self.train_config.rollout, dim=0),
                use_cache=True,
                use_varlen_inference=True,
                generation_config=self.generation_config
            )

            rollout_seq = self.tokenizer.batch_decode(rollout.sequences, skip_special_tokens=False)
            rollout_reward = self._compute_reward(rollout_seq, rollout.sequences.cpu().tolist(), data["labels"])
            del rollout
            torch.cuda.empty_cache()
        
        # Step 2: forward pass to get the prob of rollout
        rollout_seq = list(map(lambda x: re.sub(self.tokenizer.pad_token, "", x), rollout_seq))

        ## output rollout results
        if self.trainer.is_global_zero and self.last_output_steps != self.global_step and self.global_step % self.train_config.log_every_n_steps == 0:
            with open(os.path.join(self.train_config.ckpt_path, f"{self.global_step}_rollout.txt"), 'w') as f:
                cache = []
                for i in range(0, len(rollout_seq), self.train_config.rollout):
                    cache.append({"prompt": "", "responses": []})
                    cache[-1]["prompt"] = rollout_seq[i].split("<s><bot>")[0]
                    for j in range(0, self.train_config.rollout):
                        cache[-1]["responses"].append(rollout_seq[i+j].split("<s><bot>")[-1])
                f.write(json.dumps(cache, indent=4))

        rollout_seq = self.tokenizer(rollout_seq, add_special_tokens=False)
        inputs = self._generate_forward_inputs(rollout_seq.input_ids)

        outputs: CausalLMOutputWithPast = self.model(
            input_ids=inputs["input_ids"],
            cu_seqlens=inputs["cu_seqlens"],
            max_seqlen=inputs["max_seqlen"],
            return_dict=True,
            use_cache=False,
            use_gradient_checkpoint=True
        )
        logits, cu_seqlens = self._compute_logprob(outputs.logits, inputs)

        if self.ref_model is not None and self.train_config.kl_beta > 0.:
            with torch.no_grad():
                ref_outputs: CausalLMOutputWithPast = self.ref_model(
                    input_ids=inputs["input_ids"],
                    cu_seqlens=inputs["cu_seqlens"],
                    max_seqlen=inputs["max_seqlen"],
                    return_dict=True,
                    use_cache=False
                )
                ref_logits, ref_cu_seqlens = self._compute_logprob(ref_outputs.logits, inputs)
            kl_loss = torch.exp(ref_logits - logits) - (ref_logits - logits) - 1
        
        seq_len = cu_seqlens[1:] - cu_seqlens[:-1]
        advantages = rollout_reward["reward"].repeat_interleave(seq_len, dim=0)
        old_logits = logits.detach()
        coef1 = torch.exp(logits - old_logits)
        coef2 = torch.clamp(coef1, min=1-self.train_config.clip_eps, max=1+self.train_config.clip_eps)
        token_loss = -torch.min(coef1 * advantages, coef2 * advantages)
        
        if self.ref_model is not None and self.train_config.kl_beta > 0.:
            token_loss = token_loss + kl_loss * self.train_config.kl_beta

        scatter_index = torch.arange(cu_seqlens.size(0) - 1, device=self.device, dtype=torch.int64)
        scatter_index = scatter_index.repeat_interleave(seq_len, dim=0)
        final_loss = scatter(token_loss, scatter_index, dim_size=cu_seqlens.size(0) - 1, reduce="mean")

        if self.ref_model is not None and self.train_config.kl_beta > 0.:
            kl_loss = scatter(kl_loss, scatter_index, dim_size=cu_seqlens.size(0) - 1, reduce="mean")
            kl_loss = kl_loss.mean()
        else:
            kl_loss = None

        return dict(
            loss=final_loss.mean(),
            kl_loss=kl_loss,
            response_length=rollout_reward["response_length"],
            answer_reward=rollout_reward["answer_reward"],
            format_reward=rollout_reward["format_reward"],
            pass_rate=rollout_reward["pass_rate"]
        )
    
    def training_step(self, batch: Dict, batch_idx):
        outputs: Dict = self(batch)

        self.train_metrics["loss"].update(outputs["loss"])
        self.train_metrics["response_length"].update(outputs["response_length"])
        self.train_metrics["answer_reward"].update(outputs["answer_reward"])
        self.train_metrics["format_reward"].update(outputs["format_reward"])
        self.train_metrics["pass_rate"].update(outputs["pass_rate"])

        if self.ref_model is not None: self.train_metrics["kl_loss"].update(outputs["kl_loss"])

        if batch_idx % self.trainer.accumulate_grad_batches == 0:
            loss = self.train_metrics["loss"].compute()
            if self.ref_model is not None: kl_loss = self.train_metrics["kl_loss"].compute()
            response_length = self.train_metrics["response_length"].compute()
            answer_reward = self.train_metrics["answer_reward"].compute()
            format_reward = self.train_metrics["format_reward"].compute()
            pass_rate = self.train_metrics["pass_rate"].compute()

            lr = self.optimizers().optimizer.param_groups[0]["lr"]
            steps = self.trainer.global_step

            self.log_dict(
                dict(loss=loss, lr=lr, steps=steps),
                prog_bar=True,
                logger=False
            )
            self.log_dict({"grpo/loss": loss, "grpo/response_length": response_length, "grpo/answer_reward": answer_reward, "grpo/format_reward": format_reward, "grpo/pass_rate": pass_rate, "grpo/lr": lr})
            if self.ref_model is not None: self.log("grpo/kl_loss", kl_loss)

            for k, v in self.train_metrics.items():
                if isinstance(v, Metric): v.reset()

        return outputs["loss"]
    
    def on_validation_epoch_start(self):
        self.response_cache = []
        for k, v in self.eval_metrics.items():
            if isinstance(v, Metric): v.reset()

    def validation_step(self, batch: Dict, batch_idx):
        with torch.no_grad():
            rollout = self.model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=True,
                use_varlen_inference=True,
                generation_config=self.generation_config
            )
            outputs = self.tokenizer.batch_decode(rollout.sequences, skip_special_tokens=False)
        
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
        
        correct_cache = torch.tensor(correct_cache, device=self.trainer.strategy.root_device)
        label_cache = torch.ones_like(correct_cache)
        self.eval_metrics["pass@1"].update(correct_cache, label_cache)
    
    def on_validation_epoch_end(self):
        for k, v in self.eval_metrics.items():
            if isinstance(v, Metric):
                self.log(f"eval/{k}", v.compute(), sync_dist=True)

def main(train_config: argparse.Namespace):
    pl.seed_everything(train_config.seed)

    _dir = os.path.dirname(os.path.abspath(__file__))
    train_config.data_path = os.path.join(_dir, train_config.data_path)
    train_config.ckpt_path = os.path.join(_dir, train_config.ckpt_path)
    logger_path = train_config.ckpt_path if not os.path.exists("/root/tf-logs") else "/root/tf-logs"

    if not os.path.exists(train_config.ckpt_path): os.makedirs(train_config.ckpt_path)

    model_config = Sophie0Config()
    model_config.use_gradient_checkpoint = True
    
    logger = SwanLabLogger(project="Sophie0", experiment_name="reasoning_grpo", logdir=logger_path)

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
        log_every_n_steps=train_config.log_every_n_steps,
        enable_checkpointing=False,
        val_check_interval=0.5
    )

    raw_batch_size = train_config.batch_size
    if trainer.num_devices > 1:
        train_config.batch_size = train_config.batch_size // trainer.num_devices
        train_config.eval_batch_size = train_config.eval_batch_size // trainer.num_devices
    if trainer.accumulate_grad_batches > 1: 
        train_config.batch_size = train_config.batch_size // trainer.accumulate_grad_batches

    model = Sophie0ForCausalLM(model_config)
    ref_model = Sophie0ForCausalLM(model_config) if train_config.kl_beta > 0. else None
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(os.path.join(_dir, "model", "tokenizer"), use_fast=True, trust_remote_code=True, local_files_only=True)

    train_dataset = GRPODataset(tokenizer, model_config, train_config)
    eval_dataset = KKDataset(tokenizer, model_config, train_config)

    if train_config.max_steps == -1: train_config.max_steps = ((len(train_dataset) + raw_batch_size - 1) // raw_batch_size) * train_config.max_epochs

    if train_config.pretrained_ckpt_path != "": 
        _state_dict = torch.load(train_config.pretrained_ckpt_path, map_location='cpu', weights_only=True)
        model.load_state_dict(_state_dict)
        if ref_model is not None: ref_model.load_state_dict(_state_dict)

    plmodel = GRPOModule(tokenizer, model, ref_model, model_config, train_config)
    trainer.fit(
        plmodel,
        train_dataloaders=torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            num_workers=train_config.num_workers,
            collate_fn=train_dataset.process,
            persistent_workers=True,
            pin_memory=True
        ),
        val_dataloaders=torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=train_config.eval_batch_size,
            shuffle=False,
            num_workers=train_config.num_workers,
            collate_fn=eval_dataset.process,
            pin_memory=True
        )
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset Args:
    grpo_parser = parser.add_argument_group("grpo")
    grpo_parser.add_argument("--seed", type=int, default=17)
    grpo_parser.add_argument("--data_path", type=str, help="Path to the dataset dir", default="data/reasoning")
    grpo_parser.add_argument("--ckpt_path", type=str, help="Path to the checkpoint", default="result/grpo_reasoning")
    grpo_parser.add_argument("--pretrained_ckpt_path", type=str, default="", help="Path to the pretrained checkpoint")

    grpo_parser.add_argument("--batch_size", type=int, default=1)
    grpo_parser.add_argument("--eval_batch_size", type=int, default=16)
    grpo_parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients for every n batches")
    grpo_parser.add_argument("--num_workers", type=int, default=4)
    grpo_parser.add_argument("--log_every_n_steps", type=int, default=20)

    grpo_parser.add_argument("--max_steps", type=int, default=-1)
    grpo_parser.add_argument("--max_epochs", type=int, default=1)
    grpo_parser.add_argument("--save_steps", type=int, default=-1)
    grpo_parser.add_argument("--max_lr", type=float, default=5e-6, help="Maximum learning rate")
    grpo_parser.add_argument("--min_lr", type=float, default=0, help="Minimum learning rate")
    grpo_parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Ratio of steps to warm up learning rate")
    grpo_parser.add_argument("--precision", type=str, default="bf16-mixed")

    grpo_parser.add_argument("--rollout", type=int, default=16)
    grpo_parser.add_argument("--format_scale", type=float, default=0.1)
    grpo_parser.add_argument("--answer_scale", type=float, default=1.0)
    grpo_parser.add_argument("--clip_eps", type=float, default=0.2)
    grpo_parser.add_argument("--kl_beta", type=float, default=0.04)
    grpo_parser.add_argument("--length_penalty", type=float, default=0.0)

    args = parser.parse_args()
    # args.data_path="data/reasoning/grpo"
    # args.ckpt_path="result/reasoning/grpo"
    # args.pretrained_ckpt_path="/root/autodl-tmp/result/reasoning/sft/2025-05-16 16:27:44/pytorch_model.bin"
    # args.batch_size=16
    # args.accumulate_grad_batches=8
    # args.kl_beta = 0
    # args.length_penalty = 0.2

    main(args)