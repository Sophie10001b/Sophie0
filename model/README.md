Sophie0-SFT

### Introduction

Sophie0是一个从头实现的单人0.5B大语言模型项目，主要核心在于完整跑通**预训练(Pretrain)**、**监督微调(Supervised Fine-tune, SFT)**、**直接偏好优化(Direct Preference Optimization, DPO)**、以及基于 **组内相对策略优化(Group Relative Policy Optimization, GRPO)** 的显示**思维链推理**等主要流程。其中预训练阶段使用BAAI开源的多领域数据集，总数据量约11B Tokens，消耗52x4 GPU hours；微调阶段使用BAAI以及数学CoT数据在内总计9.74M行对话数据，消耗24x8 GPU hours；DPO阶段使用BAAI的偏好数据以及从LLama 3提取的英语对话数据在内总计159.3k对数据，消耗1x4 GPU hours；GRPO阶段使用Knights & Knaves 3ppl数据集以及从DeepSeek-R1提取的思维链对模型进行SFT和GRPO，SFT阶段总计有1.5k条数据，GRPO阶段总计有500条prompt，前者消耗10min x 1 GPU Times，后者消耗51x2 GPU hours.

此外，本项目进一步探讨了在下游SFT和DPO阶段完全使用变长(varlen)序列训练的可行性以及实现方式，充分利用了flash attention 2自带的`varlen attention` 和 `varlen RoPE`算子，同时也探讨了批量推理时引入的填充token对输出的影响，以及如何通过设计兼容varlen的KV Cache类直接基于Huggingface GenerationMixin接口无缝切块填充推理和无填充变长序列推理

更多内容详见[此处](https://github.com/Sophie10001b/sophie0)

### QuickStart

可通过如下方法直接调用本模型

```python
import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained("SophieA17/Sophie0-SFT", trust_remote_code=True)
tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained("SophieA17/Sophie0-SFT", trust_remote_code=True)

model = model.to(device="cuda:0", dtype=torch.bfloat16)
inputs = [
    "<s><user>Could you please introduce youself?</s>\n",
    "<s><user>Where is the best place for traveling in summer?</s>\n"
]

input_ids = tokenizer(inputs, return_tensors="pt", padding=True, padding_side="left", return_token_type_ids=False).to(model.device)

generation_config = GenerationConfig(
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    max_new_tokens=1024,
    do_sample=True,
    top_k=20,
    top_p=0.8,
    temperature=0.8,
    repeat_penalty=1.1,
    use_cache=True
)

outputs = model.generate(**input_ids, use_varlen_inference=True, generation_config=generation_config)
outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)
```