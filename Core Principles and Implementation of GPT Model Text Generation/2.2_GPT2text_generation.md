GPT2text_generation.py代码使用 PyTorch 实现了一个精简但结构完整的 **GPT-2 语言模型**。它包含了 GPT-2 的核心组件：**多头因果自注意力机制（Causal Multi-Head Self-Attention）**、**Pre-LN（前置层归一化）残差块**、**GELU 激活函数的前馈网络**，以及支持贪婪搜索（Greedy Search）和多项式采样（Sampling）的**自回归文本生成函数**。

---

### 一、 整体架构概述

整个模型模拟了 GPT-2 的标准架构，包含以下四个核心类：

1. **`GPT2SelfAttention`**：实现多头自注意力，并加入因果掩码（Causal Mask）防止模型“看到未来”的信息。
2. **`FeedForward`**：前馈神经网络，使用 GPT-2 标配的 GELU 激活函数。
3. **`GPT2Block`**：标准的 Transformer 块，采用 **Pre-LN（Pre-Layer Normalization）** 结构。
4. **`GPT2TextGenerator`**：顶层模型，负责 Token 嵌入、位置嵌入、多层 Transformer 堆叠、最终输出层（LM Head）以及自回归文本生成。

---

### 二、 核心模块详细解析

#### 1. `GPT2SelfAttention`（自注意力层）

* **QKV 一次性投影**：`self.c_attn = nn.Linear(embed_size, 3 * embed_size)`。通过一个线性层将输入直接映射到 $3 \times C$ 的维度，然后通过 `.split()` 切分出 Query ($Q$)、Key ($K$)、Value ($V$)，这比分别定义三个线性层更高效。
* **多头变形（Multi-Head Reshape）**：通过 `.view(B, T, heads, head_dim).transpose(1, 2)` 将张量形状从 `[B, T, C]` 转换为 `[B, heads, T, head_dim]`，让多头并行计算。
* **缩放点积注意力（Scaled Dot-Product Attention）**：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)V$$



代码中乘以 $\frac{1}{\sqrt{d_k}}$（即 `1.0 / (self.head_dim ** 0.5)`）以防止梯度消失或爆炸。
* **因果掩码（Causal Mask）**：`torch.tril(torch.ones(T, T))` 生成下三角矩阵，将未来时间步的位置填充为 `-inf`（经过 softmax 后权重趋近于 0），确保生成文本时只能依赖过去的上下文。

#### 2. `FeedForward`（前馈神经网络）

* 包含两个线性层，中间夹着一个 **`nn.GELU()`** 激活函数（GPT-2 相比传统 Transformer 使用 ReLU，改用了平滑的 GELU）。
* 通常隐藏层维度 `hidden_dim` 是嵌入维度 `embed_size` 的 4 倍（例如代码中的 768 $\rightarrow$ 3072）。

#### 3. `GPT2Block`（Transformer 模块）

* **Pre-LN 结构**：与原始 Transformer（Post-LN，先计算注意力再做 LayerNorm）不同，GPT-2 采用 **Pre-LN**（先做 LayerNorm，再进入 Attention/MLP，最后做残差相加）：

$$x = x + \text{attn}(\text{LayerNorm}(x))$$


$$x = x + \text{mlp}(\text{LayerNorm}(x))$$



这种结构在深层网络中梯度流动更稳定，训练更容易收敛。

#### 4. `GPT2TextGenerator`（主模型与生成逻辑）

* **Embedding 层**：
* `token_embedding`：将词汇表 ID 转换为稠密向量。
* `position_embedding`：为每个位置（0 到 `max_len`）学习一个位置编码，让模型感知词语在序列中的顺序。


* **输出层（`lm_head`）**：将最终的隐藏状态投影回词表大小（`vocab_size`），输出每个词的 **Logits**（未归一化的对数概率）。注意 `bias=False` 是现代 LLM 的常见做法，有助于减少参数量并提高稳定性。
* **`generate` 方法（自回归生成）**：
1. 接收初始词元列表 `start_token`。
2. 在循环中，每次将当前的序列输入模型，获取最后一个时间步的 logits。
3. 通过 `temperature`（温度系数）调节概率分布的平缓度（温度越高随机性越强，越低越倾向于最高概率）。
4. 支持两种选择策略：
* **贪婪搜索（Greedy Search）**：`torch.argmax` 选出概率最高的一项。
* **多项式采样（Sampling）**：`torch.multinomial` 根据概率分布随机采样，生成更有创造性的文本。


5. 将生成的 Token 追加回序列，直到达到 `max_len` 或遇到结束符 `eos_token_id`。



---

### 三、 运行与测试流程

在 `if __name__ == "__main__":` 代码块中：

1. **超参数设置**：定义了类似 GPT-2 Small 的配置（词表大小 50257，隐藏维度 768，12 个注意力头，6 层网络）。
2. **设备适配**：自动检测并挂载到 GPU (`cuda`) 或 CPU。
3. **推理测试**：传入起始 Token 序列 `[1, 345, 876]`，调用 `generate` 函数生成后续的 20 个 Token，并打印输出结果。
4. **运行结果**：
<img width="1466" height="39" alt="30d7f04b-8989-4f66-b3bd-00ccbe0d75ec" src="https://github.com/user-attachments/assets/ab10028d-7a99-4371-a6c6-5b3a91a39351" />
