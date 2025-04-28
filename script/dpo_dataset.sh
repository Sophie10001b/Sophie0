local_dir_baai="./data/dpo/baai"
local_dir_llama="./data/dpo/llama"

modelscope download --dataset 'BAAI/Infinity-Preference'\
    --include\
    train*.parquet\
    --local_dir "${local_dir_baai}"

modelscope download --dataset 'Magpie-Align/Magpie-Llama-3.1-Pro-DPO-100K-v0.1'\
    --include\
    data/train*.parquet\
    --local_dir "${local_dir_llama}"