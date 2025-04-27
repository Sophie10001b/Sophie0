import os
import glob
import pandas as pd
import multiprocessing

if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    tgt_data = "/cot"

    data_lists = glob.glob(cur_dir + tgt_data + "/**/train*.parquet", recursive=True)

    rows = 0
    seq_len = 0
    max_seq_len = 0
    turns = 0

    def count(data_path):
        local_rows = 0
        local_seq_len = 0
        local_max_seq_len = 0
        local_turns = 0

        df = pd.read_parquet(data_path)
        local_rows += len(df)

        for line in df.itertuples(index=False):
            for chat in line.conversations: local_turns += 1 if chat['from'] == 'human' else 0
            seq_len_list = [len(_['value']) for _ in line.conversations]
            local_seq_len += sum(seq_len_list)
            local_max_seq_len = max(local_max_seq_len, max(seq_len_list))
        
        return (local_rows, local_seq_len, local_max_seq_len, local_turns)

    # multiprocessing count
    if len(data_lists) > 1:
        with multiprocessing.Pool(processes=16) as pool:
            res = pool.map(count, data_lists)
    else:
        res = [count(data_lists[0])]
    
    rows = sum([_[0] for _ in res])
    seq_len = sum([_[1] for _ in res])
    max_seq_len = max([_[2] for _ in res])
    turns = sum([_[3] for _ in res])
    
    print(f"{tgt_data} Total Rows: {rows}")
    print(f"{tgt_data} Avg. Seq Length: {seq_len / rows}")
    print(f"{tgt_data} Max. Seq Length: {max_seq_len}")
    print(f"{tgt_data} Avg. Turns: {turns / rows}")