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
0: <s><user>请问你是由谁训练研发的呢？</s>
<s><bot>: 1929年10月15日,第五届世界互联网大会在浙江乌镇开幕.1949年10月15日,第五届世界互联网大会在浙江乌镇开幕.
 ...
 1997年10月15日,第五届世界互联网大会
1: <pad><pad><s><user>能否解释一下Transformer架构呢？</s>
<s><bot>的"河洛"(The Mirage of Venus)是19世纪英国数学家,数学家,数学家和建筑师协会(IAEA)于1936年1月1日宣布的英国皇家学会和皇家学会为英国皇家学会(IAEA)的前身.1947年, II.1948年,II.1955年,II.1958年...

2: <s><user>Could you please give a C++ example for quick sort?</s>
<s><bot>网科技讯 5月27日消息(记者王丽娟)中国科学院院士,中国科学院院士,中国科学院院士,中国工程院院士,国家自然科学基金委员会专家委员会副主任陈和生,1989年12月生,浙江大学数学系系主任.1994年12月生,中国科学院院士,中国科学院院士,中国科学院数学与系统科学研究院研究员,1995年12月生,中国科学院数学与系统科学研究院研究员,中国科学院数学与系统科学研究院研究员...

3: <pad><pad><pad><s><user>能否介绍一下中国的首都呢？</s>
<s><bot>1644年,丹麦人亨利·高登堡(William C. F. Walls)开始制作电影"罗马假日".1649年,乔治·高登堡在丹麦拍摄了第一部影片"罗马假日".1660年,亨利·高登堡将这部电影拍摄成一部电影拍摄成一部电影.1676年,英国电影制片厂拍摄了第一部电影"罗马假日".1682年,英国电影制片厂拍摄了第一部电影"罗马假日"...

4: <pad><s><user>Could you please tell me a joke about the weather?</s>
<s><bot>, the most famous dish of the Portuguese and the best in the country, was once a small town. The Portuguese played a key role in the history of the region, but the Portuguese were not the first to embrace it, and the Portuguese played a major role in the history of the region.
In the 16th century, Portuguese explorers and traders from Portugal, Spain, Portugal, Italy, and Britain arrived in Portugal to settle in the area...
```

若使用续写格式，即没有起始符与结束符，则生成结果如下：
```
0: <pad><pad><pad><pad>北京是中国的首都，, 2014
2014年10月29日, 2014年10月17日, 2014年10月19日, 2014年10月19日, 2014年10月19日, 2014年10月19日, 2014年10月19日, 2014年10月19日, 2014年10月19日, 2014年10月19日, 2014年10月19日, 2014年10月19日, 2014年10月19日, 2014年10月19日, 2014年10月19日, 2014年10月19日, 2014年10月19日, 2014年10月19日, 2014年10月19日...
1: There is a story for the cat and the mouse: it's a story of the human race, the human mind, and the human body. The story is told in the narrative of the story. The story is told in the narrative of the story.
The story of the human soul is told in the narrative of the story...
```
可以看见，整体效果依旧较差，存在有很多的重复以及无意义输出，可以认为仅经过pretrain阶段的语言模型并不能很好的利用训练阶段使用的庞大语料库知识

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

**Results**&emsp; SFT阶段的loss曲线如下，可以看出大约2个epoch正好接近平稳
![fig2](fig/sft_loss.png)

而在生成方面，我首先尝试了标准的对话格式，其生成结果如下：
```
0: <s><user>请问你是由谁训练研发的呢？</s>
<s><bot>作为AI语言模型，我没有个人情感，也没有情感。但是，我可以为你提供一些关于如何训练和训练神经网络的方法，但请注意，训练神经网络需要使用大量的数据和计算资源，因此我无法回答您的问题。</s>

1: <pad><pad><s><user>能否解释一下Transformer架构呢？</s>
<s><bot>Transformer架构是一种基于卷积神经网络（Transformer）架构的架构，它通过将输入序列划分为多个子序列，并通过反向传播算法（Decoder）将输入序列分解成更小的子序列。Transformer架构的核心是生成器（Transformer），它通过在输入序列上生成一个输出序列，然后通过反向传播算法来更新输入序列中的下一个元素，最后再将输入序列转换成输出序列。
Transformer架构的应用非常广泛，包括机器学习、深度学习、自然语言处理、计算机视觉和自然语言生成等领域。这些技术可以用于许多不同的应用领域，如语音识别、自然语言生成、机器翻译、语音识别和自动驾驶等。</s>

2: <s><user>Could you please give a C++ example for quick sort?</s>
<s><bot>Sure! Here's a C++ example for quick sort:

\```cpp
#include <iostream>
using namespace std;

void quickSort(int arr[], int n) {
    int n = arr[0];
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < n - i; j++) {
            if (arr[j] > arr[j + 1]) {
                cout << arr[j] << " ";
            }
            else {
                cout << arr[j] << " ";
            }
        }
    }
}

int main() {
    int arr[] = {1, 2, 3, 4, 5};
    int n = arr[0];
    cout << "Sorted array: ";
    for (int i = 0; i < n; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;

    return 0;
}
\```

This code defines a function `quickSort` that takes an array `arr` and its size `n` as parameters. It uses a `for` loop to iterate through the array and compare each element with the previous

3: <pad><pad><pad><s><user>能否介绍一下中国的首都呢？</s>
<s><bot>中国的首都是位于中国云南省和云南省交界处，是中国重要的城市。它位于中国中部，是中国最重要的国际航运中心之一，也是中国最重要的国际航空枢纽之一。</s>

4: <pad><s><user>Could you please tell me a joke about the weather?</s>
<s><bot>Sure! The weather can be a bit confusing for everyone, but there's a funny twist on it. One way to say it is "climate rain" is to say "forecast" or "snowy."</s>
```

可以看出，整体情况比仅完成预训练的模型好了不少，但在回答准确性上仍然较差。从中英对比的角度来看，英文问答会比中文问答相对更好，这可能是由于中文语料在整体训练数据中占比相对较少，不过模型基本上可以初步的对指令进行遵循，并试图依据指令作答，整体效果还是非常明显的

另外，我额外尝试对比了数学题在cot和非cot形式下的回答情况，其生成结果如下：
```
0: <pad><pad><pad><pad><pad><pad><pad><pad><s><user><s><user>Given that $b$ is a multiple of $2373$, find the greatest common divisor of $b^2 + 13b + 40$ and $b + 5$.</s>
<s><bot>Since $b$ is a multiple of $2373$, we can write $b^2 + 13b + 40 = (2373 + 40)(2373 - 13b + 40)$. Expanding the right side, we get $2373^2 - 13b + 40 = 2373^2 - 15b + 40$. Since $b$ is a multiple of $2373$, we can write $b^2 + 13b + 40 = (2373 - 15b + 40)(2373 - 13b + 40)$. Expanding the right side, we get $2373^2 - 15b + 40 = (2373^2 - 15b + 40)(2373 - 13b + 40)$. Expanding the right side, we get $2373^2 - 15b + 40 = (2373 - 15b + 40)(2373 + 15b + 40)$. Expanding the right side, we get $2373^2 - 15b + 40 = (2373 - 15b + 40)(2373 + 15b + 40)$. Expanding the right side, we get $2373^2 - 15b + 40 = (2373 - 15b + 40)(2373 + 15b + 40)$. Expanding the right side, we get $2373^2 - 15b + 40 = (2373 - 15b + 40)(2373 + 15b + 40)$. Expanding the right side, we get $2373^2 - 15b + 40 = (2373 - 15b + 40)(2373 + 15b + 40)$. Expanding the right side, we get $2373^2 - 15b + 40 = (2373 - 15b + 

1: <s><user><s><user>Given that $b$ is a multiple of $2373$, find the greatest common divisor of $b^2 + 13b + 40$ and $b + 5$. You could think it step by step.</s>
<s><bot>**
First, we need to find the greatest common divisor (GCD) of $b^2 + 13b + 40$ and $b + 5$. 

1. **Calculating $b^2 + 13b + 40$:**
   \[
   b^2 + 13b + 40 = (b + 5)^2 - 40 = b^2 + 225 - 40 = b^2 + 224
   \]

2. **Calculating $b + 5$:**
   \[
   b + 5 = 5 + 5 = 10
   \]

3. **Finding the GCD of $b^2 + 224$:**
   \[
   \text{GCD}(b^2 + 224) = \frac{b^2 + 224}{10} = \frac{10 + 224}{10} = \frac{225}{10} = 225
   \]

Thus, the greatest common divisor of $b^2 + 13b + 40$ and $b + 5$ is $\boxed{225}$.</s>
```

可以看出，cot形式输出的整体格式好了不少，但答案依旧是错的，整体还是受限于模型的训练语料规模

### DPO


### References
[^sardana2024ChinchillaOptimal]: Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws. Sardana, Nikhil, et al. ICML'24, https://openreview.net/forum?id=0bmXrtTDUu