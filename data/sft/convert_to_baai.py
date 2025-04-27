import os
import glob
import pandas as pd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    baai_data_list = glob.glob(cur_dir + "/baai/**/*.parquet", recursive=True)
    cot_data_list = glob.glob(cur_dir + "/cot/**/*.parquet", recursive=True)
    self_data_list = glob.glob(cur_dir + "/self_cognition/**/*.jsonl", recursive=True)

    # modify cot data
    def convert(messages):
        conversations = []
        for msg in messages:
            if msg['role'] == 'user':
                conversations.append({'from': 'human', 'value': msg['content']})
            elif msg['role'] == 'assistant':
                conversations.append({'from': 'gpt', 'value': msg['content']})
        return conversations

    for cot_data in cot_data_list:
        cur_df = pd.read_parquet(cot_data)
        # convert
        if 'messages' in cur_df.columns: cur_df['conversations'] = cur_df['messages'].apply(convert)
        cur_df = cur_df.drop(columns=['messages'])

        table = pa.Table.from_pandas(cur_df)
        pq.write_table(table, cot_data)

        print(f"{cot_data} converted.")
    
    # modify self_cognition data
    def self_convert(data):        
        data.response = data.response.replace("{{NAME}}", "Sophie0")
        data.response = data.response.replace("{{AUTHOR}}", "Sophie")

    self_pd = pd.read_json(self_data_list[0], lines=True)
    self_pd.apply(self_convert, axis=1)

    cache = []
    for line in self_pd.itertuples(index=False):
        conversations = [
            {"from": "human", "value": line.query},
            {"from": "gpt", "value": line.response}
        ]
        cache.append({'conversations': conversations})
    
    self_df = pd.DataFrame(cache)
    table = pa.Table.from_pandas(self_df)
    pq.write_table(table, os.path.join(os.path.dirname(self_data_list[0]), "self_cognition.parquet"))
    print(f"{self_data_list[0]} converted.")
