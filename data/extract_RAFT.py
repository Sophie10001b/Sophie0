import os
import re
import pandas as pd
import json
import sys
import torch
import transformers

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset, Dataset
from tqdm import tqdm
from itertools import chain

from torch.nn.parallel import DistributedDataParallel as DDP
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

        datas: Dataset = load_dataset("parquet", data_files=[data_path], split="train", streaming=False, trust_remote_code=True, cache_dir=HF_CACHE, num_proc=4, columns=["label", "conversations"])
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
        prompts_raw = []
        target = []
        for i, (label, conversation) in enumerate(zip(raw["label"], raw["conversations"])):
            if (conversation[0]["from"] == "user" and conversation[1]["from"] == "gpt_reasoning" and conversation[2]["from"] == "gpt_output"):
                user_input = f"<s><user>{conversation[0]['value']}</s>\n<s><bot>"
                patterns = []
                for name, solution in zip(label["names"], label["solution"]):
                    character = "knight" if solution else "knave"
                    patterns.append(rf"[\s]*\([\d]\)[\s]?{name} is a [kK]{character[1:]}")
                
                prompts_raw.append(conversation[0]['value'])
                prompts.append(user_input)
                target.append(tuple(patterns))
        
        return {
            "input_ids": prompts,
            "labels": target,
            "prompts": prompts_raw
        }
    
    def process(self, indices: list[int]):
        input_ids = self.datas.select(indices)["input_ids"]
        labels = self.datas.select(indices)["labels"]
        prompts = self.datas.select(indices)["prompts"]
        
        inputs = self.tokenizer(input_ids, return_tensors="pt", padding="longest", padding_side="left").to(torch.distributed.get_rank())
        return dict(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            labels=labels,
            prompts=prompts
        )
    
    def __getitem__(self, idx: int):
        return idx
    
    def __len__(self):
        return len(self.datas)

if __name__ == "__main__":

    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(local_rank)

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tokenizer_path = os.path.join(base_path, "model/tokenizer")
    model_path = os.path.join(base_path, "result/reasoning/sft/2025-05-16 16:27:44/pytorch_model.bin")
    data_path = os.path.join(base_path, "data/reasoning/grpo/dsr1_results_grpo.parquet")

    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=True, local_files_only=True)
    model = Sophie0ForCausalLM(Sophie0Config())

    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)

    dtype = torch.bfloat16
    model: Sophie0ForCausalLM = model.to(dtype=dtype, device=torch.distributed.get_rank())
    ddp_model = DDP(model, device_ids=[torch.distributed.get_rank()])

    generate_config = GenerationConfig(
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=2048,
        do_sample=True,
        top_k=50,
        top_p=0.8,
        temperature=0.8,
        num_beams=1,
        repeat_penalty=1.1,
        use_cache=True
    )

    batch_size = 10
    dataset = KKDataset(tokenizer, data_path)
    sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=world_size, rank=torch.distributed.get_rank())
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.process,
        sampler=sampler
    )

    shot_num = 2
    result_cache = {
        "format_correct": 0,
        "answer_correct": 0,
    }
    sample_cache = {"conversations":[]}

    bar = tqdm(dataloader, desc=f"Evaluating... Format matched: {result_cache["format_correct"]} / {len(dataloader) * shot_num}. Answer correct: {result_cache["answer_correct"]} / {len(dataloader)}.") if torch.distributed.get_rank() == 0 else None

    for _iter, inputs in enumerate(dataloader):
        if _iter > 1: break
        with torch.inference_mode():
            input_ids: torch.Tensor = inputs["input_ids"]
            input_ids = input_ids.repeat_interleave(shot_num, 0)
            attention_mask: torch.Tensor = inputs["attention_mask"].repeat_interleave(shot_num, 0)
            outputs = ddp_model.module.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                use_varlen_inference=False,
                generation_config=generate_config
            )
        
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        for i in range(0, len(outputs), shot_num):
            batch_outputs = outputs[i:i+shot_num]
            batch_ids = i // shot_num
            
            answer_correct = False
            for output in batch_outputs:
                response: str = output.split("<bot>")[-1]
                if "<think>" not in response or "</think>" not in response: continue
                if (response.count("<think>") != 1 or response.count("</think>") != 1 or response.count("</s>") != 1): continue

                result_cache["format_correct"] += 1
                reasoning = output.split("<think>")[-1].split("</think>")[0]
                answer = output.split("</think>")[-1]

                is_correct = True
                answer_patterns = inputs["labels"][batch_ids]
                for answer_pattern in answer_patterns:
                    if re.search(answer_pattern, answer) is None: is_correct = False
                
                answer_correct = answer_correct or is_correct

                # get rollout
                if is_correct:
                    sample_cache["conversations"].append([
                        {"from": "user", "value": inputs['prompts'][batch_ids]},
                        {"from": "gpt", "value": response.split("<pad>")[0]}
                    ])
            
            if answer_correct: result_cache["answer_correct"] += 1
        
        if bar is not None:
            bar.update()
            bar.set_description(f"Evaluating... Format matched: {result_cache["format_correct"]} / {len(dataset) * shot_num}. Answer correct: {result_cache["answer_correct"]} / {len(dataset)}.")
    
    if bar is not None: bar.close()
    
    all_result_cache = [None for _ in range(torch.distributed.get_world_size())]
    all_sample_cache = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(all_result_cache, result_cache)
    torch.distributed.all_gather_object(all_sample_cache, sample_cache)

    if torch.distributed.get_rank() == 0:
        tmp_result_cache = {
            "format_correct": 0,
            "answer_correct": 0,
        }
        tmp_sample_cache = {"conversations":[]}

        for cache in all_result_cache:
            for k, v in cache.items(): tmp_result_cache[k] = tmp_result_cache[k] + v
        for cache in all_sample_cache:
            tmp_sample_cache["conversations"].append(cache["conversations"])
        tmp_sample_cache["conversations"] = chain(*tmp_sample_cache["conversations"])

        result_cache = tmp_result_cache
        sample_cache = tmp_sample_cache
    
        print(f"Finished. Shot num: {shot_num}. Correct: {result_cache["answer_correct"]} | {result_cache["answer_correct"] / len(dataset):.4f}. Format: {result_cache["format_correct"]} | {result_cache["format_correct"] / len(dataset) * shot_num:.4f}.")

        current_path = os.path.dirname(os.path.abspath(__file__))
        raft_path = os.path.join(current_path, "reasoning/raft")
        if not os.path.exists(raft_path): os.makedirs(raft_path)
        df = pd.DataFrame(sample_cache)
        df.to_parquet(os.path.join(raft_path, f"raft_{shot_num}.parquet"))
    
    torch.distributed.destroy_process_group()