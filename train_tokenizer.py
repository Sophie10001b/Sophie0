import os
import re
import glob
import json
import pandas as pd

import tokenizers
import tokenizers.decoders
import tokenizers.pre_tokenizers
import tokenizers.normalizers
import tokenizers.processors
import transformers

from typing import List
from tokenizers import Tokenizer, NormalizedString, PreTokenizedString, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

if __name__ == "__main__":
    _dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(_dir, "data")

    data_lists = glob.glob(data_path + "/**/*.parquet", recursive=True)

    # def data_generator():
    #     for corpus_dir in data_lists:
    #         corpus = pd.read_parquet(corpus_dir)
    #         if "literature_emotion/chinese" in corpus_dir: corpus = corpus[:int(len(corpus) * 0.6)]
    #         for row in corpus.itertuples(index=False):
    #             yield row.text
    
    # tokenizer = Tokenizer(BPE())
    
    # tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence([
    #     tokenizers.pre_tokenizers.Split(
    #         pattern=Regex("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"),
    #         behavior="isolated"
    #     ),
    #     tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False, trim_offsets=False),
    # ])
    # tokenizer.normalizer = tokenizers.normalizers.NFC()
    # tokenizer.post_processor = tokenizers.processors.ByteLevel(add_prefix_space=False, use_regex=False, trim_offsets=False)
    # tokenizer.decoder = tokenizers.decoders.ByteLevel(add_prefix_space=False, use_regex=False, trim_offsets=False)

    # # for _data in data_generator():
    # #     _res = tokenizer.normalizer.normalize_str(_data)
    # #     _res = tokenizer.pre_tokenizer.pre_tokenize_str(_res)
    # #     pass

    # trainer = BpeTrainer(
    #     vocab_size=65536,
    #     min_frequency=2,
    #     show_progress=True,
    #     special_tokens=["<s>", "</s>", "<unk>", "<pad>", "<mask>", "<sep>", "<think>", "</think>", "<prompt>", "<user>", "<bot>"],
    #     max_token_length=16,
    #     initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet()
    # )

    # tokenizer.train_from_iterator(data_generator(), trainer=trainer)
    # tokenizer.save(os.path.join(_dir, "model", "tokenizer.json"))

    # convert to AutoTokenizer
    # tokenizer = Tokenizer.from_file(os.path.join(_dir, "model", "tokenizer.json"))
    # tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    # tokenizer.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>", "pad_token": "<pad>", "mask_token": "<mask>", "sep_token": "<sep>", "additional_special_tokens": ["<think>", "</think>", "<prompt>", "<user>", "<bot>"]})
    # tokenizer.save_pretrained(os.path.join(_dir, "tokenizer"))