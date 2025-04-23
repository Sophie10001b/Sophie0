import os
import glob
import pandas
import json
import torch
import torch.nn as nn
import transformers
import lightning as pl

from itertools import chain
from datasets import Dataset, load_dataset
from lightning import Trainer, LightningDataModule, LightningModule
from transformers import AutoTokenizer
from flash_attn.bert_padding import unpad_input

from model.modeling_sophie0 import Sophie0ForCausalLM
from model.configuration_sophie0 import Sophie0Config

class PretrainDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        config: Sophie0Config,
        tokenizer: AutoTokenizer,
        max_seqlen: int,
        max_token_per_batch: int,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.data_path = data_path
        self.config = config
        self.tokenizer = tokenizer
        self.max_seqlen = max_seqlen
        self.max_token_per_batch = max_token_per_batch

        data_files = glob.glob(data_path + "/**/*.parquet", recursive=True)
        datas: Dataset = load_dataset("parquet", data_files=data_files, split="train", streaming=False, trust_remote_code=True, columns=["text"])
        datas = datas.shuffle(seed=kwargs.pop("seed", 17))

        # pre-chunk
        datas = datas.map(
            self._preprocess,
            batched=True,
            batch_size=1000,
            num_proc=1,
            remove_columns=datas.column_names
        )
    
    def _preprocess(self, raw):
        texts = [self.tokenizer.bos_token + _ + self.tokenizer.eos_token for _ in raw['text']]
        outputs = self.tokenizer(texts, add_special_tokens=False)['input_ids']
        
        texts = list(chain(outputs))
        pass
    
    def process(self, indices: list[int]):
        texts = [self.tokenizer.bos_token + _["text"] + self.tokenizer.eos_token for _ in self.datas.select(indices)]
        outputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="longest"
        )

        varlen_texts, _, cu_seqlens, max_seqlen, _ = unpad_input(outputs["input_ids"][:, :-1].unsqueeze(-1), attention_mask=outputs["attention_mask"][:, :-1])
        varlen_labels, _, _, _, _ = unpad_input(outputs["input_ids"][:, 1:].unsqueeze(-1), attention_mask=outputs["attention_mask"][:, 1:])
        
        return dict(
            input_ids=varlen_texts.squeeze(-1),
            labels=varlen_labels.squeeze(-1),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen
        )

    def __getitem__(self, idx: int):
        return idx
    
    def __len__(self):
        return len(self.datas)

class PretrainDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        config: Sophie0Config,
        tokenizer: AutoTokenizer,
        max_seqlen: int=2048,
        max_token_per_batch: int=1048576,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.data_path = data_path
        self.config = config
        self.tokenizer = tokenizer
        self.max_seqlen = max_seqlen
        self.max_token_per_batch = max_token_per_batch

        self.data = PretrainDataset(data_path, config, tokenizer, max_seqlen, max_token_per_batch)
    
    def setup(self, stage):
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.data,
            batch_size=self.max_token_per_batch,
            shuffle=True,
            num_workers=0,
            collate_fn=self.data.process,
            pin_memory=True
        )