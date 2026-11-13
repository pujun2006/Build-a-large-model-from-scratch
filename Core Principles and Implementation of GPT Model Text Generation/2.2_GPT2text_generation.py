import torch
import torch.nn as nn
import torch.nn.functional as F


class GPT2SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(GPT2SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "embed_size 必须能被 heads 整除"

        # 修正：投影层输入输出均为 embed_size
        self.c_attn = nn.Linear(embed_size, 3 * embed_size)
        self.c_proj = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        B, T, C = x.size()  # Batch size, Sequence length, Embedding size

        # 一次性计算 Q, K, V
        q, k, v = self.c_attn(x).split(self.embed_size, dim=2)

        # 转换维度为 [B, heads, T, head_dim]
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        # 计算 Scaled Dot-Product Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))

        # 修正：生成在对应设备上的因果掩码 (Causal Mask)
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        # 拼接多头输出
        out = att @ v  # [B, heads, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.c_proj(out)


class FeedForward(nn.Module):
    def __init__(self, embed_size, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_size)
        # 修正：GPT-2 规范采用 GELU 激活函数
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class GPT2Block(nn.Module):
    def __init__(self, embed_size, heads, hidden_dim, dropout):
        super(GPT2Block, self).__init__()
        self.ln_1 = nn.LayerNorm(embed_size)
        self.attn = GPT2SelfAttention(embed_size, heads)
        self.ln_2 = nn.LayerNorm(embed_size)
        self.mlp = FeedForward(embed_size, hidden_dim, dropout)

    def forward(self, x):
        # 修正：标准的 GPT-2 Pre-LN 结构
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2TextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, hidden_dim, max_len, dropout=0.1):
        super(GPT2TextGenerator, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_len, embed_size)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [GPT2Block(embed_size, heads, hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_size)  # GPT-2 最终输出前的 LayerNorm
        self.lm_head = nn.Linear(embed_size, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        # Token 嵌入 + 位置嵌入
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(pos)
        x = self.dropout(tok_emb + pos_emb)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, start_token, max_len, temperature=1.0, sample=False, eos_token_id=None):
        self.eval()  # 切换到评估模式，关闭 Dropout
        generated = list(start_token)

        for _ in range(max_len - len(start_token)):
            x = torch.tensor([generated], device=next(self.parameters()).device)
            # 控制上下文不超过模型支持的最大长度
            x_cond = x if x.size(1) <= max_len else x[:, -max_len:]

            logits = self.forward(x_cond)
            logits = logits[:, -1, :] / temperature

            # 支持多项式概率采样或贪婪搜索
            if sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = torch.argmax(logits, dim=-1).item()

            generated.append(next_token)
            if eos_token_id is not None and next_token == eos_token_id:
                break

        return generated


# --- 执行与测试 ---
if __name__ == "__main__":
    vocab_size = 50257
    embed_size = 768
    num_layers = 6  # 为简化示范调整层数
    heads = 12
    hidden_dim = 3072
    max_len = 1024
    dropout = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2TextGenerator(vocab_size, embed_size, num_layers, heads, hidden_dim, max_len, dropout).to(device)

    start_tokens = [1, 345, 876]

    # 模拟文本生成
    generated_sequence = model.generate(start_tokens, max_len=20, temperature=0.8, sample=True)
    print("生成的序列 ID:", generated_sequence)

