nproc=4

seed=17
data_path="data/dpo"
ckpt_path="result/dpo"
pretrained_ckpt_path="result/sft/pytorch_model.bin"

max_steps=-1
max_epochs=1
save_steps=-1

max_seqlen=2048
max_token_per_batch=524288
accumulate_grad_batches=16

max_lr=5e-7
min_lr=0

python dpo.py \
    --data_path $data_path \
    --ckpt_path $ckpt_path \
    --pretrained_ckpt_path "${pretrained_ckpt_path}" \
    --seed $seed \
    --max_steps $max_steps \
    --max_epochs $max_epochs \
    --save_steps $save_steps \
    --max_seqlen $max_seqlen \
    --max_token_per_batch $max_token_per_batch \
    --accumulate_grad_batches $accumulate_grad_batches \
    --max_lr $max_lr \
    --min_lr $min_lr \
    --beta 0.1