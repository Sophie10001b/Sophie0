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

最后在120G内存下使用64core机器于25min内完成词表构建

### Model
模型选择了标准的Transformer Decoder结构，参考现有相似规模的相关工作，将$d_{model}$设为1024，$d_{ff}$设为4096，$n_{layer}$设为28，$n_{head}$设为16，GQA分组数量设置为8，即每两个query head对应于一个key value head。Attention, RoPE以及SwiGLU的实现直接套用[Flash Attention 2](https://github.com/Dao-AILab/flash-attention)自带的实现方式，其中RoPE base直接设置为1M以避免后续在长上下文下额外调整base的情况。最终模型参数总量正好来到了0.5B的规模。由于整体规模有限，vocab embedding直接占据了总学习参数量的13%左右，因此进一步共享了模型的embedding层与lm head投影层以提高中间参数的总占比。

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

### References
[^sardana2024ChinchillaOptimal]: Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws. Sardana, Nikhil, et al. ICML'24, https://openreview.net/forum?id=0bmXrtTDUu
