import os
import sys
import torch
import transformers

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, GenerationConfig
from flash_attn.bert_padding import unpad_input

from model.configuration_sophie0 import Sophie0Config
from model.modeling_sophie0 import Sophie0ForCausalLM

if __name__ == "__main__":

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tokenizer_path = os.path.join(base_path, "model/tokenizer")
    model_path = os.path.join(base_path, "result/dpo/pytorch_model.bin")

    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=True, local_files_only=True)
    model = Sophie0ForCausalLM(Sophie0Config())

    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)

    device = "cuda:0"
    dtype = torch.bfloat16
    model: Sophie0ForCausalLM = model.to(dtype=dtype, device=device)

    generate_config = GenerationConfig(
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=256,
        do_sample=True,
        top_k=20,
        top_p=0.7,
        temperature=0.8,
        num_beams=1,
        repeat_penalty=1.1,
        use_cache=True
    )

    prompt = [
        "<s><user>请问你是由谁训练研发的呢？</s>\n<s><bot>",
        "<s><user>能否解释一下Transformer架构呢？</s>\n<s><bot>",
        "<s><user>Could you please give a C++ example for quick sort?</s>\n<s><bot>",
        "<s><user>能否介绍一下中国的首都呢？</s>\n<s><bot>",
        "<s><user>Could you please tell me a joke about the weather?</s>\n<s><bot>"
    ]
    inputs = tokenizer(prompt, return_tensors="pt", padding="longest", padding_side="left")
    
    flatten_input, _, cu_seqlens, max_seqlen, _ = unpad_input(inputs.input_ids.unsqueeze(-1), inputs.attention_mask)
    
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        outputs = model.generate(
            input_ids=inputs.input_ids.to(device),
            attention_mask=inputs.attention_mask.to(device),
            use_cache=True,
            use_varlen_inference=True,
            generation_config=generate_config
        )
    
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    for i, output in enumerate(outputs):
        print(f"{i}: {output}")