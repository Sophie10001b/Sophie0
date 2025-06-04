import os
import re
import json
import glob
import multiprocessing
import pandas as pd

from tqdm import tqdm

def check_and_convert(raw: dict):
    user_input = raw['quiz']
    name_list = raw['names']
    solution = raw['solution']

    prompt, answer = {}, []
    prompt["names"] = name_list
    prompt["solution"] = solution

    reasoning_incomplete = 0
    answer_incorrect = 0

    for output in raw['outputs']:
        output_reasoning: str = output['reasoning']
        output_result: str = output['result']

        # check reasoning is complete or not
        if not output_reasoning.endswith(('.\n', '\n', '.')):
            reasoning_incomplete += 1
            continue

        # check answer is correct or not
        is_correct = True
        for i, (name, _answer) in enumerate(zip(name_list, solution)):
            character = "knight" if _answer else "knave"
            pattern = rf"[\s]*\([\d]\)[\s]?{name} is a [kK]{character[1:]}"
            is_match = re.search(pattern, output_result)

            if not is_match:
                is_correct = False
                answer_incorrect += 1
                break
        
        if not is_correct: continue

        # save the remaining data
        answer.append([
            {"from": "user", "value": user_input},
            {"from": "gpt_reasoning", "value": output_reasoning},
            {"from": "gpt_output", "value": raw["solution_text_format"]}
        ])
    return (reasoning_incomplete, answer_incorrect, prompt, answer)

if __name__ == "__main__":
    cur_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(cur_path, "dsr1_results.jsonl")

    with open(data_path, 'r') as f:
        raw_data = json.load(f)
    
    results = []
    
    for data in tqdm(raw_data):
        results.append(check_and_convert(data))

    formated_results = []
    reasoning_incomplete, answer_incorrect = 0, 0
    for i, data in enumerate(results):
        incomplete, incorrect, prompt, answer = data
        reasoning_incomplete += incomplete
        answer_incorrect += incorrect
        for _answer in answer:
            formated_results.append(
                {
                    "id": len(formated_results),
                    "label": prompt,
                    "conversations": _answer
                }
            )
    
    print(f"Reasoning incomplete: {reasoning_incomplete} / {len(results)}")
    print(f"Answer incorrect: {answer_incorrect} / {len(results) * 3}")
    
    df = pd.DataFrame(formated_results[:int(len(formated_results) / 2)])
    df.to_parquet(os.path.join(cur_path, "dsr1_results_sft.parquet"))

    df = pd.DataFrame(formated_results[int(len(formated_results) / 2):])
    df.to_parquet(os.path.join(cur_path, "dsr1_results_grpo.parquet"))