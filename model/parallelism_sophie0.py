import torch
from modeling_sophie0 import Sophie0ForCausalLM
from torch.distributed._composable.fsdp import MixedPrecisionPolicy
from torch.distributed._composable.fsdp.fully_shard import fully_shard
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)

def parallelize_setting(model: Sophie0ForCausalLM, device_mesh: DeviceMesh) -> Sophie0ForCausalLM:
    dp_mesh = device_mesh["data_parallel"]
    tp_mesh = device_mesh["tensor_parallel"]

    # no tp parallel setting
    
    # examples from:
    # https://github.com/Lightning-AI/pytorch-lightning/blob/master/examples/pytorch/tensor_parallel/parallelism.py
    if dp_mesh.size() > 1:
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
        for layer_id, block in enumerate(model.model.layers):
            reshard_after_forward = int(layer_id) < len(model.model.layers) - 1
            fully_shard(
                block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward
            )
        
        fully_shard(model.model, **fsdp_config)
        fully_shard(model, **fsdp_config)
    return model