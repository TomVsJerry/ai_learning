# Transformer 学习材料

这里是后续排学习路线用的底稿，不是已经排好的课表。时间和顺序可以改；过关标准先固定，避免只读过、对不上公式。

材料来自对 Vaswani 等人 2017 年论文的整理，数字以原文表格为准。论文：[arXiv:1706.03762](https://arxiv.org/abs/1706.03762)。

## 文件

| 文件 | 用途 |
| --- | --- |
| [00-neural-network-framework.md](00-neural-network-framework.md) | 人工神经网络从头到尾的一条流水线：同一套权重判断两杯饮料喜不喜欢 |
| [../../examples/mnist-digits/README.md](../../examples/mnist-digits/README.md) | 可运行的手写数字项目：784 个亮度，10 个数字的百分比 |
| [01-matrix-and-dot-product.md](01-matrix-and-dot-product.md) | 阶段 1：用奶茶口味和步行方向讲向量、点积、转置和矩阵形状 |
| [02-softmax.md](02-softmax.md) | 阶段 2：用匹配分讲指数、归一化、饱和，以及为什么要除以 √d_k |
| [03-linear-relu-embedding.md](03-linear-relu-embedding.md) | 阶段 3：线性层的底分、ReLU 把负数清零、词编号查表得到向量 |
| [04-residual-layernorm.md](04-residual-layernorm.md) | 阶段 4：在原向量上加修改，再按这一杯自己的平均和标准差拉回同一尺度 |
| [transformer-and-neural-networks.md](transformer-and-neural-networks.md) | Transformer 是神经网络的一种接法，对照全连接、循环网络和卷积网络 |
| [prerequisites.md](prerequisites.md) | 读模型之前要会的技能、形状自测，以及读到后面再补的内容 |
| [attention-is-all-you-need.md](attention-is-all-you-need.md) | 按论文章节写的导读：架构、注意力、训练、实验结果 |

对话里的画布仍可打开，但不在仓库中。以本目录的 Markdown 为准。

## 规划底稿

同一层的阶段可以并行。勾选表示该项过关标准已经达到。备注留给之后填时间、材料和顺序。

| 状态 | 阶段 | 过关时要能做到 | 对照 | 备注 |
| --- | --- | --- | --- | --- |
| [ ] | 1. 矩阵与点积 | 由行列数写出乘积形状；说明点积大表示方向更接近 | [细读](01-matrix-and-dot-product.md)、[过关标准](prerequisites.md#stage-matrix) | |
| [ ] | 2. Softmax | 写出 exp 再归一化；说明输入差很大时梯度接近 0 | [细读](02-softmax.md)、[过关标准](prerequisites.md#stage-softmax) | |
| [ ] | 3. 线性层、ReLU、词嵌入 | 写出 `y = xW + b` 和 `max(0, ·)`；说明 token 查表得到向量 | [细读](03-linear-relu-embedding.md)、[过关标准](prerequisites.md#stage-linear) | |
| [ ] | 4. 残差与 LayerNorm | 写出 `LayerNorm(x + Sublayer(x))`，并说明各子层宽度必须相同 | [细读](04-residual-layernorm.md)、[过关标准](prerequisites.md#stage-residual) | |
| [ ] | 5. 编码器–解码器 | 说明训练时解码器看到已给定前缀，预测下一个词 | [前置：序列模型](prerequisites.md#stage-seq2seq) | |
| [ ] | 6. 加权求和式注意力 | 说明权重来自查询和键的匹配，输出是值向量的加权和 | [前置：注意力定义](prerequisites.md#stage-attention) | |
| [ ] | 7. 形状自测 | 独立推出 n=4、8 头时从 `X` 到 `W^O` 的每一步形状 | [前置：形状](prerequisites.md#stage-shapes) | |
| [ ] | 8. 论文第 3 节 | 对照导读讲清编码器、三种注意力、掩码和前馈 | [导读：架构](attention-is-all-you-need.md#paper-arch)、[注意力](attention-is-all-you-need.md#paper-attn) | |
| [ ] | 9. 位置编码 | 用和角公式接受 `PE(pos+k)` 可由 `PE(pos)` 线性表示 | [前置：3.5](prerequisites.md#stage-pos)、[导读：位置](attention-is-all-you-need.md#paper-arch) | |
| [ ] | 10. 第 4 节动机 | 说明 RNN 为何在样本内串行，以及路径长度 `O(n)` 与 `O(1)` 的差别 | [前置：循环](prerequisites.md#stage-rnn)、[导读：动机](attention-is-all-you-need.md#paper-motive) | |
| [ ] | 11. 第 5 节训练 | 读懂学习率公式，并说明标签平滑对困惑度和 BLEU 的影响 | [前置：训练](prerequisites.md#stage-train)、[导读：训练](attention-is-all-you-need.md#paper-train) | |
| [ ] | 12. 第 6 节实验 | 分清开发集 25.8 与测试集 28.4；知道 BLEU 和束搜索不是训练损失 | [前置：实验](prerequisites.md#stage-eval)、[导读：实验](attention-is-all-you-need.md#paper-results) | |

实现时再看 [batch、稳定 softmax 和掩码](prerequisites.md#stage-impl)。LSTM 门控公式、RoPE、FlashAttention、KV cache、RLHF 放在 [先不展开](prerequisites.md#stage-later)。

依赖关系：

```text
矩阵与点积 ─┐
            ├─ 线性层与嵌入 ─┬─ 编码器–解码器 ─ 注意力加权和 ─┐
Softmax ────┘               └─ 残差与 LayerNorm ────────────┴─ 论文第 3 节
```
