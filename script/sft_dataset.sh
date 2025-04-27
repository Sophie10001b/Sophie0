local_dir_baai="./data/sft/baai"
local_dir_cot="./data/sft/cot"
local_dir_self="./data/sft/self_cognition"

modelscope download --dataset 'BAAI/Infinity-Instruct'\
    --include\
    '7M/*.parquet'\
    'Gen/*.parquet'\
    --local_dir "${local_dir_baai}"

modelscope download --dataset 'OmniData/NuminaMath-CoT'\
    --include\
    'data/train*.parquet'\
    --local_dir "${local_dir_cot}"

modelscope download --dataset 'swift/self-cognition'\
    --include\
    '*.jsonl'\
    --local_dir "${local_dir_self}"

python ./data/sft/convert_to_baai.py