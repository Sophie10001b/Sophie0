nproc=4

seed=17
data_path="data/pretrain"
ckpt_path="result/pretrain"

max_steps=50
max_epochs=-1
save_steps=10

max_seqlen=2048
max_token_per_batch=524288
accumulate_grad_batches=8

max_lr=2e-4
min_lr=0

python pretrain.py \
    --data_path $data_path \
    --ckpt_path $ckpt_path \
    --seed $seed \
    --max_steps $max_steps \
    --max_epochs $max_epochs \
    --save_steps $save_steps \
    --max_seqlen $max_seqlen \
    --max_token_per_batch $max_token_per_batch \
    --accumulate_grad_batches $accumulate_grad_batches \
    --max_lr $max_lr \
    --min_lr $min_lr