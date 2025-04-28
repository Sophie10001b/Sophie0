local_dir_baai="./data/dpo/baai"

modelscope download --dataset 'BAAI/Infinity-Preference'\
    --include\
    train-00000-of-00001.parquet\
    --local_dir "${local_dir_baai}"