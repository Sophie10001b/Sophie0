## Sophie0 -- 单人0.5B Toy LLM项目

### Tokenizer
参考Qwen 2.5，使用Huggingface Tokenizers库提供的BPE实现，加上Qwen 2.5使用的NFC Normalization + Regex Split + ByteLevel进行pre tokenization，得到BBPE词表，总大小为65536。其中tokenizer数据集来源如下，均直接从预训练数据集中抽取固定大小的子集，最终得到了中文:英文约2:5的词表构建语料:

| Path | Size |
| --- | --- |
| accommodation_catering_hotel/english/high/rank_00726.parquet | 513M |
| news_media/chinese/high/rank_00082.parquet | 318M |
| news_media/english/high/rank_01332.parquet | 544M |
| mathematics_statistics/english/high/rank_01082.parquet | 519M |
| computer_programming_code/english/high/rank_00864.parquet | 444M |
| literature_emotion/chinese/high/rank_00063.parquet | 905M (仅使用前60%) |

最后在120G内存下使用64core机器于25min内完成词表构建。为了测试tokenizer的效果，我选择了词表构建中使用的数据集以及其它未参与词表构建的语料进行了相应的测试，使用压缩率(原始语料长度/分词后语料长度)作为评价指标：

| Path | Char Counts | Sophie0 Comp. | Qwen2.5 Comp. |
| --- | --- | --- | --- |
| news_media/chinese/high/rank_00082.parquet | 0.18B | 1.8940 | 1.6205 |
| news_media/english/high/rank_01332.parquet | 0.92B | 4.6466 | 4.7232 |
| mathematics_statistics/english/high/rank_01082.parquet | 1.14B | 3.6269 | 3.2916 |
| computer_programming_code/english/high/rank_00864.parquet | 0.86B | 4.2720 | 4.2283 |
| technology_scientific_research/chinese/high/rank_00123.parquet | 0.52B | 1.7518 | 1.6499 |
| technology_scientific_research/english/high/rank_01466.parquet | 0.86B | 4.0325 | 3.8732 |

可以看出，在以上的预训练语料中，Sophie0在使用一致的normalizer和pre-tokenizer策略的情况下整体压缩率略优于Qwen2.5的原生tokenizer，猜测可能是因为测试集没有包含除中英外的多语种或符号数据集，Qwen2.5的原生大词表可能在更加OOD的语料上更有优势

### Model
模型方面，Sophie0选择了标准的Transformer Decoder结构，参考现有相似规模的相关工作，将主要参数设置如下：

| Name | Value |
| --- | --- |
| hidden_size | 1024 |
| intermediate_size | 4096 |
| num_layers | 28 |
| num_q_heads | 16 |
| num_kv_heads | 8 |
| RoPE base | 1e6 |
| Param. | 0.5B |

其中Attention, RoPE以及SwiGLU的实现直接套用[Flash Attention 2](https://github.com/Dao-AILab/flash-attention)自带的实现方式，RoPE base直接调整至1M以免去在下游重新缩放base的需要。最终模型参数总量正好来到了0.5B的规模。由于整体规模有限，vocab embedding直接占据了总学习参数量的13%左右，因此进一步共享了模型的embedding层与lm head投影层以提高中间参数的总占比。

对于Attention部分，由于Sophie0在预训练阶段参考相关工作使用了Sequence Packing将带有BOS和EOS的文档统一拼接并切分为2,048 token序列长度，但在下游SFT & RL阶段则难以使用类似技术统一序列长度。考虑到计算开销的优化，Sophie0专门基于输入序列的总维度分别实现了标准attention + varlen attention，从而保证下游微调阶段的有效吞吐量。另外，考虑到推理阶段对各类beam search和KV Cache维护的支持，varlen attention仅用于训练阶段。

### Pretrain
**Datasets**&emsp; 预训练使用BAAI发布的[IndustryCorpus2](https://www.modelscope.cn/datasets/BAAI/IndustryCorpus2)，直接使用modelscope的API选择部分质量分类为高的子集进行下载，最终得到了总计26GB，中英比例约2:5的预训练语料，其具体数据分布如下:

| Path | Size |
| --- | --- |
| accommodation_catering_hotel | 1.08GB |
| artificial_intelligence_machine_learning | 1.84GB |
| biomedicine | 2.14GB |
| computer_programming_code | 0.78GB |
| computer_communication | 1.83GB |
| current_affairs_government_administration | 3.83GB |
| game | 0.93GB |
| film_entertainment | 2.07GB |
| mathematics_statistics | 3.68GB |
| news_media | 3.45GB |
| technology_scientific_research | 2.64GB |
| tourism_geography | 1.94GB |

经过处理后总token数量约11B左右，与ICML'24上的文章[^sardana2024ChinchillaOptimal]内建议的最低token-per-param (20 tokens/param) 一致，即达到该比例后，继续增加训练数据的增益与所带啦的额外训练开销的比值基本保持不变，可以认为达到了相对最优的trade-off

**Settings**&emsp; 预训练阶段使用AutoDL中的4张vGPU-32GB完成训练，其中每张标定fp16算力为103 TFLOPs，Compute Capability为8.9，论坛内推测为32GB版本的4080s。并行策略直接使用pytorch lightning的原生FSDP设置，单batch设置跑0.5M tokens，序列长度固定2,048，平均下来每张卡单步更新需要跑到64 batch size，经过测试将梯度累积设定为8，即单卡每步跑8个序列，梯度累积8步，总batch size为256。运行下来4张卡的显存与CUDA基本完全吃满，单步forward + backward大概1.09s，估算下来每一步更新大概需要接近9s，整体跑完大致花费52h，开销约370RMB

**Results**&emsp; 首先是预训练loss，由于训练时仅统计了rank0的batch loss而没有进行多卡reduce，因此loss曲线结果的波动会比正常更大，仅供参考
![fig1](fig/pretrain_loss.png)

随后是对话，由于预训练阶段模型没有系统的学习如何遵循特定的指令以及格式，因此它无法很好的关联上下文，以及预测结束符`</s>`，导致模型生成的结果基本为预训练阶段见过的语料：
```

```

### SFT
**Datasets**&emsp; Sophie0在微调阶段同样使用了由BAAI发布的[Infinity-Instruct](https://www.modelscope.cn/datasets/BAAI/Infinity-Instruct)，使用其中的7M大小基础数据集 + 额外的Gen数据集(也就是对话数据集)构建为最终的SFT语料。此外，为了让模型初步学习到CoT模式以及自我认知信息，SFT阶段语料库中同时混入了[NuminaMath-CoT](https://www.modelscope.cn/datasets/OmniData/NuminaMath-CoT)数据让模型学习数学相关的推理知识。最终的用于CoT的数据组成如下，其中序列长度仅统计分词前的原始长度：

| Path | Rows | Seq Length | Turns of Chat |
| --- | --- | --- | --- |
| Infinity-Instruct 7M | 7.44M | 1.34k (max 2.68M) | 1.23 |
| Infinity-Instruct Gen | 1.45M | 3.03k (max 354.76k) | 1.06 |
| NuminaMath-CoT | 0.85M | 1.42k (max 10.39k) | 1.00 |
| swift-self_cognition | 108 | 120.85 | 1.00 |

**Settings**&emsp; 微调阶段由于不同数据长度差异较大，Sophie0直接启用了flash attention的`varlen`算子，为了让每batch的最大token数量可控，同时也最大限度保证每batch内的数据不会浪费，Sophie0在预处理阶段通过以下步骤进行数据切分：

1. 在每一行完整对话数据内以`user-assistant`的QA-pair形式遍历单论对话并拼接，若分词后的token长度已经达到每batch的token数量上限，则将对话进行切分，后续轮次的对话切分为新的轮次。最终切分为`Chat1 {{QA1, QA2, ..., QAi}, {QAi+1, ...}, ...}, Chat2, ...`的形式
2. 将最内层截断的`{QA1, QA2, ..., QAi}`单轮/多轮对话数据作为一个batch(batch = 1)喂入模型，若出现当前对话数据长度较少的情况，则会查询后一个独立对话数据内的切分情况，将前后多项独立的对话数据拼接为一个batch(batch > 1)

举例来说，假如现在有两个独立的多轮对话数据：
```
Chat1:
    Q1: Could you please introduce yourself?
    A1: Surely, I am a useful AI assistant create by ..., please no doubt to ask me anything.
    Q2: Thank you. Bye.
    A2: Bye.

Chat2:
    Q1: How's the weather today?
    A1: It's sunny.
```
那么得到的batch数据将会如下：
```
batch1:
    data1: <s><user>Could you please introduce yourself?</s>\n<s><bot>Surely, I am a useful AI assistant create by ..., please no doubt to ask me anything.</s>

batch2:
    data1: <s><user>Thank you. Bye.</s>\n<s><bot>Bye.</s>
    data2: <s><user>How's the weather today?</s>\n<s><bot>It's sunny.</s>
```

在varlen情况下，所有batch的首个维度都被flatten至`(batch size * sequence length)`，因此在实际加载数据的时候可以直接在数据预处理阶段按照batch最大token数进行预切分，之后直接设置batch size为1来生成varlen格式的数据即可

不过，虽然已经有了上面的切分策略来保证每个batch的数据量能够尽量逼近但不超过固定上界，但在实际SFT时少部分情况仍然出现了单卡OOM的情况导致FSDP训练失败，因此最后直接扩展至8卡vGPU-32GB进行训练，每步更新的最大token总数为0.5M，梯度累积8步，总计训练2 epoch，接近16k steps，耗时正好约24h，开销约320RMB

**Results**&emsp; loss曲线如下
![fig2](fig/sft_loss.png)

### DPO


### References
[^sardana2024ChinchillaOptimal]: Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws. Sardana, Nikhil, et al. ICML'24, https://openreview.net/forum?id=0bmXrtTDUu