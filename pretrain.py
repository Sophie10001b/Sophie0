import os
import pandas
import json
import torch
import torch.nn as nn
import transformers
import lightning as pl

from tokenizers import Tokenizer
from lightning import Trainer, LightningDataModule, LightningModule

from model.modeling_sophie0 import Sophie0ForCausalLM
from model.configuration_sophie0 import Sophie0Config

if __name__ == '__main__':
    config = Sophie0Config()
    model = Sophie0ForCausalLM(config)

    model.save_pretrained()