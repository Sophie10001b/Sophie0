import os
import re
import json
import sys
import torch
import transformers

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset, Dataset
from tqdm import tqdm
from flash_attn.bert_padding import unpad_input

from transformers import AutoTokenizer, GenerationConfig
from model.configuration_sophie0 import Sophie0Config
from model.modeling_sophie0 import Sophie0ForCausalLM

# HF_CACHE = None
# HF_CACHE = os.environ["HF_HOME"]
HF_CACHE = "/root/autodl-tmp/hf_cache"
os.environ["HF_HOME"] = HF_CACHE

class KKDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer: AutoTokenizer, data_path: str, **kwargs):
        super().__init__(**kwargs)

        self.tokenizer = tokenizer
        self.data_path = data_path

        datas: Dataset = load_dataset("json", data_files=[data_path], split="train", streaming=False, trust_remote_code=True, cache_dir=HF_CACHE, num_proc=4)
        self.datas = datas.map(
            self.convert_kk_dataset,
            batched=True,
            batch_size=1000,
            num_proc=4,
            remove_columns=datas.column_names,
            load_from_cache_file=False
        )

    def convert_kk_dataset(self, raw):
        system_prompt = """
        I will provide some quiz about knights-and-knaves. Please point out who is the knight and who is the knave briefly and clearly and output them in a specific format. Please remind that DO NOT return any further explanation, and just return the formated answer.

        EXAMPLE INPUT: 
        A very special island is inhabited only by knights and knaves. Knights always tell the truth, and knaves always lie. You meet 2 inhabitants: Oliver, and Ethan. Oliver told you that Oliver is a knight or Ethan is a knave. In a statement by Ethan: "Oliver is a knight". So who is a knight and who is a knave?

        EXAMPLE OUTPUT:
        (1) Oliver is a knight
        (2) Ethan is a knight

        Now here is the question:

        """

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
        
        inputs = tokenizer(input_ids, return_tensors="pt", padding="longest", padding_side="left")
        return dict(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            labels=labels
        )
    
    def __getitem__(self, idx: int):
        return idx
    
    def __len__(self):
        return len(self.datas)

if __name__ == "__main__":

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tokenizer_path = os.path.join(base_path, "model/tokenizer")
    model_path = os.path.join(base_path, "result/reasoning/sft/2025-05-16 16:27:44/pytorch_model.bin")
    data_path = os.path.join(base_path, "data/reasoning/people3_num100.jsonl")

    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=True, local_files_only=True)
    model = Sophie0ForCausalLM(Sophie0Config())

    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)

    device = "cuda:0"
    dtype = torch.bfloat16
    model: Sophie0ForCausalLM = model.to(dtype=dtype, device=device)

    generate_config = GenerationConfig(
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

    batch_size = 1
    dataset = KKDataset(tokenizer, data_path)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.process
    )

    shot_num = 10
    result_cache = {
        "format_correct": 0,
        "reasoning_complete": 0,
        "answer_correct": 0,
    }

    tqdm_dataloader = tqdm(dataloader, desc=f"Evaluating... Format matched: {result_cache["reasoning_complete"]} / {len(dataset) * shot_num}. Answer correct: {result_cache["answer_correct"]} / {len(dataset)}.")
    for inputs in tqdm_dataloader:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids: torch.Tensor = inputs["input_ids"]
            input_ids = input_ids.repeat_interleave(shot_num, 0)
            attention_mask: torch.Tensor = inputs["attention_mask"].repeat_interleave(shot_num, 0)
            outputs = model.generate(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                use_cache=True,
                use_varlen_inference=False,
                generation_config=generate_config
            )
        
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        for i in range(0, len(outputs), batch_size):
            batch_outputs = outputs[i:i+batch_size]
            batch_ids = i // shot_num
            
            answer_correct = False
            for output in batch_outputs:
                response = output.split("<bot>")[-1]
                if "<think>" not in response or "</think>" not in response: continue

                result_cache["format_correct"] += 1
                reasoning = output.split("<think>")[-1].split("</think>")[0]
                answer = output.split("</think>")[-1]

                if "</s>" not in answer: continue
                result_cache["reasoning_complete"] += 1
                answer = answer.split("</s>")[0]

                is_correct = True
                answer_patterns = inputs["labels"][batch_ids]
                for answer_pattern in answer_patterns:
                    if re.search(answer_pattern, answer) is None: is_correct = False
                
                answer_correct = answer_correct or is_correct
            
            if answer_correct: result_cache["answer_correct"] += 1
        tqdm_dataloader.set_description(f"Evaluating... Format matched: {result_cache["reasoning_complete"]} / {len(dataset) * shot_num}. Answer correct: {result_cache["answer_correct"]} / {len(dataset)}.")
    
    print(f"Finished. Shot num: {shot_num}. Correct: {result_cache["answer_correct"]} | {result_cache["answer_correct"] / len(dataset):.4f}. Format: {result_cache["format_correct"]} | {result_cache["format_correct"] / len(dataset):.4f}. Reasoning complete: {result_cache["reasoning_complete"]} | {result_cache["reasoning_complete"] / len(dataset):.4f}.")
