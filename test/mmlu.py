import os
import re
import numpy as np
import pandas as pd
import glob
import sys
import torch
import transformers

# sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightning.pytorch import seed_everything
from transformers import AutoTokenizer, GenerationConfig
from datasets import Dataset, load_dataset
from tqdm.std import tqdm

from model.configuration_sophie0 import Sophie0Config
from model.modeling_sophie0 import Sophie0ForCausalLM

HF_CACHE = os.environ["HF_HOME"]
# HF_CACHE = "/root/autodl-tmp/hf_cache"
# os.environ["HF_HOME"] = HF_CACHE

CHOICE = {0: "(A)", 1: "(B)", 2: "(C)", 3: "(D)"}
CHOICE_STR = {"A": "(A)", "B": "(B)", "C": "(C)", "D": "(D)"}

class MMLUDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer: AutoTokenizer, data_path: str, data_subject: str):
        super().__init__()
        
        dataset = pd.read_csv(data_path, header=None)
        self.tokenizer = tokenizer
        self.questions = []
        self.answers = []

        for data in dataset.itertuples(index=False):
            question, answer = mmlu_process(data_subject, data[0], [data[1], data[2], data[3], data[4]], data[5])
            if question is not None:
                self.questions.append(question)
                self.answers.append(answer)
        
        self.questions = np.array(self.questions, dtype=object)
        self.answers = np.array(self.answers, dtype=object)
    
    def process(self, indices: list[int]):
        questions = self.questions[indices].tolist()
        questions = self.tokenizer(questions, return_tensors="pt", padding="longest", padding_side="left").input_ids

        return dict(
            questions=questions,
            answers=self.answers[indices].tolist()
        )
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx: int):
        return idx

def mmlu_process(subject: str, question: str, choices: list[str], answer: str | int):
    if answer not in [0, 1, 2, 3] and answer not in ['A', 'B', 'C', 'D']: return None, None

    prompt = f"<s><user>The following is a question in {subject.replace('_', ' ')}, please select the correct answer from (A), (B), (C), (D). Here is a instance about the answer format:\n"
    instance = "Q:\nWhich one is the biggest number?\n(A): 1\n(B): 2\n(C): 3\n(D): 4\nA:\nThe answer is (C).\n"
    question = "Then here is the question:\n" + question + "\n" + "\n".join([f"{_choice}: {_answer}" for _choice, _answer in zip(["(A)", "(B)", "(C)", "(D)"], choices)]) + "\nPlease directly select the correct answer according to the answer format.</s>\n<s><bot>"

    questions = prompt + instance + question
    answers = CHOICE.get(answer) if isinstance(answer, int) else CHOICE_STR.get(answer)
    
    return questions, answers

if __name__ == "__main__":
    seed_everything(17)

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tokenizer_path = os.path.join(base_path, "model/tokenizer")
    model_path = os.path.join(base_path, "result/sft/finish/pytorch_model.bin")
    data_path = os.path.join(base_path, "data/benchmark/mmlu_test")

    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=True, local_files_only=True)
    model = Sophie0ForCausalLM(Sophie0Config())

    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)

    device = "cuda:0"
    dtype = torch.bfloat16
    model = model.to(device=device, dtype=dtype)

    generate_config = GenerationConfig(
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=512,
        do_sample=True,
        top_k=20,
        top_p=0.8,
        temperature=0.7,
        num_beams=1,
        repeat_penalty=1.1,
        use_cache=True
    )

    data_list = glob.glob(data_path + "/*.csv", recursive=True)
    benchmark_dict = {}
    judge = re.compile(r'\(([A-D])\)')

    for data_path in data_list:
        domain = data_path.split("/")[-1].split(".")[0]
        dataset = MMLUDataset(tokenizer, data_path, domain)
        benchmark_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=16,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=dataset.process
        )

        benchmark_dict[domain] = 0.0

        for i, data in enumerate(tqdm(benchmark_loader)):
            question, answer = data['questions'], data['answers']
            question = question.to(device)

            outputs = model.generate(
                question,
                generate_config
            )
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)

            for j, output in enumerate(outputs):
                prompt, response = output.split("<bot>")
                bot_answer = re.findall(judge, response)
                bot_answer = int(bot_answer[-1] == answer) if bot_answer else 0
                benchmark_dict[domain] += bot_answer

                if i == 0 and j == 0: print(f"\n{domain}:\n{prompt}\n{response}")

        benchmark_dict[domain] /= len(dataset)
        print(f"{domain}: {benchmark_dict[domain]:.4f}")
    
    print("Final Results:")
    for domain in benchmark_dict:
        print(f"{domain}: {benchmark_dict[domain]:.4f}")