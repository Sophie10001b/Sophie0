local_dir="./data/sft"

modelscope download --dataset 'BAAI/Infinity-Instruct'\
    --include\
    '7M/*.parquet'\
    'Gen/*.parquet'\
    --local_dir "${local_dir}"