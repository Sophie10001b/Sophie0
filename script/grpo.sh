nproc=2

seed=17
data_path="data/reasoning/grpo"
ckpt_path="result/reasoning/grpo"
pretrained_ckpt_path="/root/autodl-tmp/result/reasoning/sft/2025-05-16 16:27:44/pytorch_model.bin"

max_steps=-1
max_epochs=5
save_steps=-1

batch_size=16
accumulate_grad_batches=8

rollout=16

max_lr=5e-6
min_lr=0

python reasoning_grpo.py \
    --data_path $data_path \
    --ckpt_path $ckpt_path \
    --pretrained_ckpt_path "${pretrained_ckpt_path}" \
    --seed $seed \
    --max_steps $max_steps \
    --max_epochs $max_epochs \
    --save_steps $save_steps \
    --batch_size $batch_size \
    --accumulate_grad_batches $accumulate_grad_batches \
    --max_lr $max_lr \
    --min_lr $min_lr \
    --rollout $rollout \
    --log_every_n_steps 2 \
    --kl_beta 0.04 \
    --length_penalty 0.0