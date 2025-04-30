import os
import glob
import sys
import torch
import transformers

# sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightning.pytorch import seed_everything
from transformers import AutoTokenizer, GenerationConfig
from datasets import Dataset, load_dataset
from tqdm.std import trange

from model.configuration_sophie0 import Sophie0Config
from model.modeling_sophie0 import Sophie0ForCausalLM

HF_CACHE = os.environ["HF_HOME"]
# HF_CACHE = "/root/autodl-tmp/hf_cache"
# os.environ["HF_HOME"] = HF_CACHE

CHOICE = {0: "(A)", 1: "(B)", 2: "(C)", 3: "(D)"}

def mmlu_process(raw):
    questions, answers = [], []
    for subject, question, choices, answer in zip(raw["subject"], raw["question"], raw["choices"], raw["answer"]):
        if answer not in [0, 1, 2, 3]: continue

        prompt = f"<s><user>The following is a question in {subject.replace('_', ' ')}, please select the correct answer from (A), (B), (C), (D). Here is a instance about the answer format:\n"
        instance = "Q:\nWhich one is the biggest number?\n(A): 1\n(B): 2\n(C): 3\n(D): 4\nA:\nBecause 4 > 3 > 2 > 1, so the correct answer is (C).\n"
        question = "Then here is the question:\n" + question + "\n".join([f"{_choice}: {_answer}" for _choice, _answer in zip(["(A)", "(B)", "(C)", "(D)"], choices)]) + "\nPlease select the correct answer according to the answer format.</s>\n<s><bot>"

        questions.append(prompt + instance + question)
        answers.append(CHOICE.get(answer))
    
    return {
        "question": questions,
        "answer": answers
    }

if __name__ == "__main__":
    seed_everything(17)

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tokenizer_path = os.path.join(base_path, "model/tokenizer")
    model_path = os.path.join(base_path, "result/sft/finish/pytorch_model.bin")
    data_path = os.path.join(base_path, "data/benchmark/mmlu")

    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=True, local_files_only=True)
    model = Sophie0ForCausalLM(Sophie0Config())

    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)

    # load benchmark
    benchmark_lists = glob.glob(data_path + "/*.parquet", recursive=True)
    benchmark: Dataset = load_dataset("parquet", data_files=benchmark_lists, split="train", streaming=False, trust_remote_code=True, columns=["subject", "question", "choices", "answer"], cache_dir=HF_CACHE, num_proc=4)
    benchmark.shuffle(seed=17)

    benchmark = benchmark.map(
        mmlu_process,
        batched=True,
        batch_size=5000,
        num_proc=1,
        remove_columns=benchmark.column_names,
        load_from_cache_file=False,
        cache_file_name=os.path.join(HF_CACHE, "parquet/benchmark/mmlu_map.cache")
    )

    benchmark_loader = torch.utils.data.DataLoader(
        benchmark,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    for i, data in zip(trange(benchmark_loader), benchmark_loader):
        pass