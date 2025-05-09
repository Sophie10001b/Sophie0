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
<s><bot>Since $b$ is a multiple of $2373$, we can write $b^2 + 13b + 40 = (2373 + 40)(2373 - 13b + 40)$. Expanding the right side, we get $2373^2 - 13b + 40 = 2373^2 - 15b + 40$. Since $b$ is a multiple of $2373$, we can write $b^2 + 13b + 40 = (2373 - 15b + 40)(2373 - 13b + 40)$. Expanding the right side, we get $2373^2 - 15b + 40 = (2373^2 - 15b + 40)(2373 - 13b + 40)$...

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
**Datasets**&emsp; 在DPO阶段，Sophie0使用了BAAI的[Infinity-Preference](https://www.modelscope.cn/datasets/BAAI/Infinity-Preference)数据集以及Magpie-Align的[Llama3.1_DPO](https://huggingface.co/datasets/Magpie-Align/Magpie-Llama-3.1-Pro-DPO-100K-v0.1)数据集，前者包含有59.3k对偏好数据，后者则直接基于Llama 3.1 70B Instruct采样了约100k对偏好数据，均为单轮问答

**Settings**&emsp; 参考SFT阶段的varlen切分，DPO阶段同样使用varlen算子，并保证每一batch中以(prompt + max(正例，负例))计算的总token数上界不超过0.25M。DPO阶段仅使用4张vGPU-32G进行训练，因此梯度累积扩大为16步，即每一步有16k个token均分至4张GPU。DPO阶段为避免过拟合与训练稳定，参考已有的工作与开源项目，学习率上界设置为5e-7，训练1个epoch，更新约380步，总耗时约1.2h，开销约8.2RMB

此外，考虑到已有工作中提到的DPO训练时正负样例选择概率同时下降的问题，参考[此篇总结](https://zhuanlan.zhihu.com/p/698852522)以及另一篇工作[^liu2024RPO]，Sophie0采用了类似于RPO的损失项：
$$
\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l)\sim D}\left[\log \sigma \left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right) + \beta\log\pi_\theta(y_w|x)\right]
$$
也就是将正例的SFT损失加入至原始的偏好损失，从而约束$\pi_\theta$的正例输出概率上升。为了简化调参，该损失的倍率直接套用DPO的$\beta$

**Results**&emsp; 首先是DPO训练期间的loss曲线与reward曲线

<figure>
<img src="fig/dpo_padding/chosen_reward.png" height=150/>
<img src="fig/dpo_padding/reject_reward.png" height=150/>
<img src="fig/dpo_padding/dpo_loss.png" height=150/>
<img src="fig/dpo_padding/sft_loss.png" height=150/>
<img src="fig/dpo_padding/chosen_win.png" height=150/>
</figure>

可以发现，由于SFT损失项的约束，Sophie0在DPO过程中并没有出现明显的正例概率下降问题，即`chosen_reward`维持在了0附近，也就是说正例的选择概率与$\pi_{ref}$类似，对应的`chosen_win`也正好保持在了50%附近。而负例的reward则稳定下降，整体的偏好损失正常下降，而作为约束项的SFT loss则缓慢上升。整体曲线趋势与之前的一些工作相似。由于DPO阶段仅训练了1个epoch，因此也不需要考虑在后续epoch中的过拟合问题

另外值得注意的是，DPO阶段对batch size的要求不低，Sophie0在初始尝试的时候使用了约0.15M tokens的batch size，此时偏好loss的曲线出现明显震荡，最终模型的正常生成能力也被破坏，因此需要足够多对正负例样本来引导模型在每一步的更新

除去loss曲线外，经过DPO阶段后模型的输出结果如下：
```
0: <s><user>请问你是由谁训练研发的呢？</s>
<s><bot>你好！我是基于深度学习的人工智能模型，由OpenAI开发，旨在开发能够理解、生成和生成自然语言文本的智能客服系统。以下是我为您提供的训练计划：
1. 数据预处理：首先，需要对数据进行清洗、分词、去除停用词、词干提取、词形还原等操作，以便后续的分析和标注。
2. 模型训练：使用TensorFlow或PyTorch等深度学习框架进行模型训练，以便模型能够理解和生成自然语言文本。
3. 模型评估：使用测试集对模型进行评估，评估模型的准确性和泛化能力。
4. 应用：将训练好的模型应用于实际应用中，如语音识别、机器翻译、推荐系统等。
5. 用户测试：将测试结果反馈给用户，并根据反馈不断优化模型。
6. 持续改进：根据反馈不断优化模型，不断优化模型，提高其准确性和
1: <s><user>能否解释一下Transformer架构呢？</s>
<s><bot>Transformer（Generative Adversarial Networks，GPT）是计算机视觉和自然语言处理领域中的一个重要技术，用于处理图像和视频。它通过生成模型来生成高质量的图像，并通过学习生成模型来生成高质量的图像。
Transformer的架构如下：
1. 生成模型：GAN（Generative Adversarial Networks）是生成器，它由多个层（前馈神经网络）组成，这些层通过生成网络（如卷积神经网络）来生成图像。GAN在生成图像方面表现非常出色，能够生成高质量的图像。
2. 训练：GAN在生成图像时会进行一些微调，如裁剪、裁剪、缩放等，以提高图像质量。GAN在生成图像时会进行一些微调，如裁剪、缩放、缩放、缩放等，以提高图像质量。
3. 生成：GAN在生成图像时会进行一些微调�
2: <s><user>Could you please give a C++ example for quick sort?</s>
<s><bot>Certainly! Here's a C++ example for quick sort:

```cpp
#include <iostream>
#include <vector>

using namespace std;

int main() {
    int n;
    cout << "Enter a positive integer: ";
    cin >> n;
    for (int i = 0; i < n; i++) {
        cout << i << " ";
    }
    cout << endl;
    return 0;
}
```

In this code, we define a function `cache` that takes a positive integer `n` as input. Inside the function, we initialize an empty vector `cout` to store the elements of the array `n`. We then use a `for` loop to iterate `n` times, updating the `i` value of `n` with the current element `i` in the vector. Finally, we return the `cout` vector.

In the `main` function, we declare an integer `n` and assign it the value of `n` to it. We then call the `cache` function with `n` as the argument and store the returned vector in the `cache` vector.

When you run this program, it will output: `200
3: <s><user>能否介绍一下中国的首都呢？</s>
<s><bot>中国首都位于中国的首都，是中国政治、经济、文化、科技、科技、文化、体育和文化中心。它是中国最重要的城市之一，也是世界著名的历史文化名城之一。
北京位于中国的首都，是中国重要的国际贸易中心，也是中国重要的国际贸易中心。北京是中国重要的国际贸易中心，也是中国最重要的国际贸易中心之一。
北京是中国的首都，也是中国最重要的国际贸易中心之一。它位于中国的首都，也是中国重要的国际贸易中心之一。
北京的历史可以追溯到16世纪，当时，中国的首都是北京。在19世纪，北京成为了一个重要的国际贸易中心，并成为了中国最重要的国际贸易中心之一。
除了北京以外，北京还是中国重要的国际贸易中心之一。它位于中国的首都，也是中国重要的国际贸易中心之一。</s>
4: <s><user>Could you please tell me a joke about the weather?</s>
<s><bot>Why did the weather change the weather? Because it was so cold that it was like a cold breeze.</s>
```

整体上来看，相较于SFT后的模型，DPO对小模型的增益似乎并不明显，可能需要使用专门的偏好数据集进行验证

### GRPO + Reasoning
*Comming soon, 由于模型规模较小，预计使用一半的reasoning数据集先进行SFT以对齐格式*


### 变长推理
**Background**&emsp; 在单人用户对话中，LLM可能仅需要处理batch = 1的情况，但在跑benchmark或多用户场景，对batch inference的支持能够极大的提升GPU的利用率。一般来说，batch inference仅需要对输入的token进行左padding即可，但在LLM主要阶段全部使用padding-free策略进行训练的情况下，模型会难以处理很少见到的<pad> token，导致注意力无法正确关注到正文内容。即使在将pad token替换为eos token后仍然难以缓解该问题。

作为示例，模型的输入选择了一些短问题 + 长段的数学CoT，对应的batch inference结果如下：
```
0: <pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><s><user>能否解释一下Transformer架构呢？</s>
<s><bot>**Respondue:**

**Issue:** **Issue:** **Issue:**...
1: <pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><s><user>Could you please give a C++ example for quick sort?</s>
<s><bot>**Issue: Ensure the Invalid Code (PAT)**

**Issue: Easier to C++**

**Issue: Easier to C++**

**Issue: Easier to C++**
...
2: <pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><s><user>Could you please tell me a joke about the weather?</s>
<s><bot>Certainly, here's a list of common concerns:

1. **Temperature Issues**:
   - **Temperature Issues**: Toxic (high) and temperature (high).
   ...
3: <s><user><s><user>Given that $b$ is a multiple of $2373$, find the greatest common divisor of $b^2 + 13b + 40$ and $b + 5$. You could think it step by step.</s>
<s><bot>To determine the greatest common divisor (GCD) of $b^2 + 13b + 40$ and $b + 5$, we follow these steps:

1. **Identify the GCD of $b^2 + 13b + 40$:**
   - The prime factorization of $b^2 + 13b + 40$ is $b^2 + 13b + 40$.
   - Therefore, the GCD of $b^2 + 13b + 40$ and $b + 5$ is $b^2 + 13b + 40$.

2. **Express $b^2 + 13b + 40$ in terms of $b$:**
   - We start by expressing $b^2 + 13b + 40$ in terms of $b$ using the difference of squares formula:
     \[
     b^2 + 13b + 40 = (b + 5)(b - 2)
     \]
   - We then apply the difference of squares formula:
     \[
     (b + 5)(b - 2) = b^2 + 2b + 2
     \]
   - Rearrange to isolate $b^2 + 2b + 2$:
     \[
     b^2 + 2b + 2 = b^2 + 2b + 2 - b^2 - 2b + 2 = b^2 + 2b - 2b + 1
     \]
   - Further simplification gives:
     \[
     b^2 + 2b - 2b + 1 = 1
     \]

3. **Factorize $b^2 + 13b + 40$:**
   - We now have:
     \[
     b^2 + 13b + 40 = 1 + 13b + 40
     \]
   - To find the GCD, we need to find the values of $b$ that make the expression $1 + 13b + 40$ a multiple of $3$.

4. **Solve the equation:**
   ...
```

可以发现，在<pad>占据主要输入的情况下，模型的输出内容受到了明显影响，仅有不包含<pad>的数学CoT回答仍然保持正常。

将tokenizer的<pad>替换为结束符，对应的输出结果如下：
```
0: </s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s><s><user>能否解释一下Transformer架构呢？</s>
<s><bot>提创意的「�家。年迈时，</s>
1: </s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s><s><user>Could you please give a C++ example for quick sort?</s>
<s><bot>茙�，</s>
2: </s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s></s><s><user>Could you please tell me a joke about the weather?</s>
<s><bot>治育活２

新新饮用等社交软件。�和。�和，</s>
```

可以发现，模型虽然能够正确预测结束符，但是输出内容同样受到了影响。

为了解决这个问题，Sophie0在Huggingface的GenerateMixin，Cache框架下额外实现了兼容varlen inference的流程。作为测试，我选择了相同的生成参数，仅将与varlen inference关联的`use_varlen_inference`打开，对应的生成结果如下：
```
0: <pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><s><user>能否解释一下Transformer架构呢？</s>
<s><bot>Transformer架构（Generative Adversarial Networks，GAN）是一种基于Transformer架构的神经网络架构，它通过一个卷积神经网络（GAN）来生成输出、预测和生成输出。GAN架构通过一个卷积层（Convolution）层，将输入的输入转换为输入输出，再通过一个卷积层来生成输出。GAN架构的核心是损失函数，它将输入的输出映射到输入层，然后通过损失函数将输出映射到输出层，最终输出输出。
GAN架构的基本思想是通过一个卷积层来生成输出，并将输出映射到输出层。GAN架构的核心是损失函数，它将输出映射到输出层，然后通过损失函数将输出映射到输出层，最终输出输出。
Transformer架构在图像识别、语音识别、自然语言处理、推荐系统等多个领域中发挥着重要作用。</s>
1: <pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><s><user>Could you please give a C++ example for quick sort?</s>
<s><bot>Certainly! The C++ example for quick sort is as follows:

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

// Function to sort elements of a vector
vector<int> sort(vector<int>& arr) {
    for (int i = 0; i < arr.size(); i++) {
        for (int j = 0; j < arr[i].length; j++) {
            if (arr[j] > arr[i]) {
                arr[j] = arr[i];
                arr[i] = arr[j];
            }
        }
    }
}

int main() {
    vector<int> arr = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    sort(arr);
    return 0;
}
```

In this example, we define a function `sort` that takes an array (`arr`) and an integer (`arr[i]`) as input. Inside the function, we use the `std::sort` function to sort the elements of the array. The `sort` function takes three parameters: `arr`, `arr`, and `arr[i]`. 

The `sort` function takes three parameters: `i`, `j`, and `arr[i]`. It compares the elements at indices `i` and `j` in the array and assigns the corresponding values to `arr[i]` and `arr[j]`. The `sort` function is then used to sort the array in ascending order.

By using this code, you can easily sort any given array in ascending order.</s>
2: <pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><s><user>Could you please tell me a joke about the weather?</s>
<s><bot>Why did the weather go to ruin the town? Because it had a very cold winter, the weather was a constant.</s>
3: <s><user><s><user>Given that $b$ is a multiple of $2373$, find the greatest common divisor of $b^2 + 13b + 40$ and $b + 5$. You could think it step by step.</s>
<s><bot>To determine the greatest common divisor (gcd) of \(b^2 + 13b + 40\) and \(b + 5\):

1. **Identify the GCD of the two numbers:**
   - The greatest common divisor (GCD) of \(b^2 + 13b + 40\) and \(b + 5\) is given by \(b^2 + 13b + 40\).
   - Here, \(b = 2373\).
   - Thus, the GCD is \(2373\).

2. **Apply the Euclidean Algorithm:**
   - Apply the Euclidean algorithm to find the GCD:
     \[
     \text{GCD}(2373, 2373) = 2373
     \]
   - Calculate \(2373\):
     \[
     2373 = 2 \times 3 \times 7 \times 13
     \]
     \[
     2373 = 6 \times 7 \times 13
     \]
     \[
     2373 = 429 \times 13
     \]
     \[
     2373 = 1199 \times 13
     \]
     \[
     1199 = 13 \times 13
     \]
     \[
     1199 = 13 \times 23
     \]
     ...
```

可以发现，varlen inference保证了模型对batch中短prompt输出的正确处理，证明了其有效性

**Implementation**&emsp; 为了兼容Huggingface的主框架，varlen inference的外部输入需与原始框架完全一致，而对应的序列索引等信息则需要在推理时实时生成。其主要的改动点集中在以下几个地方：

- `prepare_inputs_for_generation`: 在该函数中，Sophie0需要生成当前步中对应query的`cu_seqlens`，并在最开始的prefill阶段将prompt转换为padding-free形式(直接使用flash_attention的相关函数)
- `VarlenCache`: 在该类中，Sophie0需要维护varlen形式的KV Cache以及KV的`cu_seqlens`，同时负责将当前新的KV插入到之前的Cache对应的位置中
- `Attention`: 在注意力层，Sophie0使用与训练时一致的varlen attention算子，不过需要注意query和KV对应着不同的序列索引
- `Output`: 在得到模型的当前步输出后，需要通过`cu_seqlens`提取出每个序列最后的token，作为当前步生成的新token

通过以上步骤，Sophie0即可使用varlen attention算子以padding-free形式处理不定长数据的batch inference

### References
[^sardana2024ChinchillaOptimal]: Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws. Sardana, Nikhil, et al. ICML'24.

[^liu2024RPO]: Provably Mitigating Overoptimization in RLHF: Your SFT Loss is Implicitly an Adversarial Regularizer. Zhihan Liu, et al. NeurIPS'24.