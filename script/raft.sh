nproc=2

seed=17
data_path="data/reasoning/raft"
ckpt_path="result/reasoning/raft"
pretrained_ckpt_path="/root/autodl-tmp/result/reasoning/sft/2025-05-16 16:27:44/pytorch_model.bin"

max_steps=-1
max_epochs=3
save_steps=-1

batch_size=32
eval_batch_size=50
accumulate_grad_batches=8

max_lr=5e-6
min_lr=0

python reasoning_raft.py \
    --data_path $data_path \
    --ckpt_path $ckpt_path \
    --pretrained_ckpt_path "${pretrained_ckpt_path}" \
    --seed $seed \
    --max_steps $max_steps \
    --max_epochs $max_epochs \
    --save_steps $save_steps \
    --batch_size $batch_size \
    --eval_batch_size $eval_batch_size \
    --max_lr $max_lr \
    --min_lr $min_lr