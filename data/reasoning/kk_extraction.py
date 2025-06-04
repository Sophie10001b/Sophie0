import os
import json
import glob
import multiprocessing

from tqdm import tqdm
from openai import OpenAI
from datasets import Dataset, load_dataset

def preprocess(raw):
    return {
        "quiz": raw["quiz"],
        "names": raw["names"],
        "solution": raw["solution"],
        "solution_text_format": raw["solution_text_format"]
    }

def extract_reasoning_from_api(raw):
    system_prompt = """
    The user will provide some quiz about knights-and-knaves. Please point out who is the knight and who is the knave briefly and clearly and output them in a specific format. Please remind that DO NOT return any further explanation, and just return the formated answer.

    EXAMPLE INPUT: 
    A very special island is inhabited only by knights and knaves. Knights always tell the truth, and knaves always lie. You meet 2 inhabitants: Oliver, and Ethan. Oliver told you that Oliver is a knight or Ethan is a knave. In a statement by Ethan: "Oliver is a knight". So who is a knight and who is a knave?

    EXAMPLE OUTPUT:
    (1) Oliver is a knight
    (2) Ethan is a knight
    """
    
    client = OpenAI(
        api_key=os.environ.get("ARK_API_KEY"), 
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )

    res_dict = dict(
        quiz=raw['quiz'],
        names=raw['names'],
        solution=raw['solution'],
        solution_text_format=raw['solution_text_format'],
        outputs=[]
    )
    for _ in range(3):
        completion = client.chat.completions.create(
            model="deepseek-r1-250120",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw['quiz']}
            ]
        )
        result_content = completion.choices[0].message.content
        reasoning_content = completion.choices[0].message.reasoning_content
        res_dict['outputs'].append(
            dict(reasoning=reasoning_content, result=result_content)
        )
    
    return res_dict
    

if __name__ == "__main__":
    base_path = os.path.abspath(__file__)
    for _ in range(3): base_path = os.path.dirname(base_path)

    os.system(API[:-1])
    os.system("echo $ARK_API_KEY")

    data_files = glob.glob(os.path.join(base_path, "data/reasoning/people3_num1000.jsonl"), recursive=True)
    datas: Dataset = load_dataset("json", data_files=data_files, split="train", streaming=False, trust_remote_code=True, num_proc=4)

    # pre-process
    datas = datas.map(
        preprocess,
        batched=True,
        batch_size=5000,
        num_proc=4,
        remove_columns=datas.column_names,
        load_from_cache_file=True
    )

    datas = datas.select(range(20))

    pbar = tqdm(total=len(datas))
    pbar.set_description("Calling DeepSeek-R1 API...")
    update = lambda *args: pbar.update()
    with multiprocessing.Pool(processes=4) as pool:
        api_res = [pool.apply_async(extract_reasoning_from_api, args=(x,), callback=update) for x in datas]
        results = [_.get() for _ in api_res]
    
    pool.close()
    pool.join()
    
    current_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_path, "dsr1_results.jsonl"), 'w') as f:
        f.write(json.dumps(results, indent=4))