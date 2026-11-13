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

        # GPT-2 规范：一次性投影生成 Q, K, V
        self.c_attn = nn.Linear(embed_size, 3 * embed_size)
        self.c_proj = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        B, T, C = x.size()

        # 投影并拆分为 Q, K, V -> [B, T, C]
        q, k, v = self.c_attn(x).split(self.embed_size, dim=2)

        # 转换为多头维度 -> [B, heads, T, head_dim]
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        # 缩放点积注意力
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))

        # 修正：将 Mask 创建在与输入相同的设备上
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        out = att @ v  # [B, heads, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.c_proj(out)


class FeedForward(nn.Module):
    def __init__(self, embed_size, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_size)
        self.act = nn.GELU()  # 修正：GPT-2 标准 GELU
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
        # 修正：GPT-2 标准 Pre-LN 结构
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
        self.ln_f = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(pos)
        x = self.dropout(tok_emb + pos_emb)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def greedy_search(self, start_token, max_len, eos_token_id=None):
        self.eval()
        generated = list(start_token)

        for _ in range(max_len - len(start_token)):
            x = torch.tensor([generated], device=next(self.parameters()).device)
            logits = self.forward(x)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()

            generated.append(next_token)
            if eos_token_id is not None and next_token == eos_token_id:
                break

        return generated

    @torch.no_grad()
    def beam_search(self, start_token, max_len, beam_width=3, eos_token_id=None):
        self.eval()
        device = next(self.parameters()).device

        # 候选池格式：(sequence, log_score)
        beams = [(list(start_token), 0.0)]
        completed_beams = []

        for _ in range(max_len - len(start_token)):
            all_candidates = []

            for seq, score in beams:
                # 若已到达终止符，转移至完成池
                if eos_token_id is not None and seq[-1] == eos_token_id:
                    completed_beams.append((seq, score))
                    continue

                x = torch.tensor([seq], device=device)
                logits = self.forward(x)
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1)

                top_k_probs, top_k_indices = torch.topk(log_probs, beam_width, dim=-1)

                for i in range(beam_width):
                    next_token = top_k_indices[0, i].item()
                    new_score = score + top_k_probs[0, i].item()
                    all_candidates.append((seq + [next_token], new_score))

            if not all_candidates:
                break

            # 筛选分数最高的前 beam_width 个候选
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            beams = ordered[:beam_width]
            # 在 beam_search 方法中添加：
            print(f"当前轮次保留的 {beam_width} 条路径：", [b[0] for b in beams])

        # 汇总最终候选（加上长度归一化防止偏向短句子）
        all_final = completed_beams + beams
        best_seq = max(all_final, key=lambda x: x[1] / len(x[0]))[0]
        return best_seq


# --- 测试运行 ---
if __name__ == "__main__":
    vocab_size = 50257
    embed_size = 768
    num_layers = 4  # 简易测试层数
    heads = 12
    hidden_dim = 3072
    max_len = 50
    dropout = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpt2_model = GPT2TextGenerator(vocab_size, embed_size, num_layers, heads, hidden_dim, max_len, dropout).to(device)

    start_token = [1, 345, 876]
    eos_id = 50256  # 假设 EOS Token ID

    # Greedy Search
    greedy_res = gpt2_model.greedy_search(start_token, max_len=10, eos_token_id=eos_id)
    print("Greedy Search 结果:", greedy_res)

    # Beam Search
    beam_res = gpt2_model.beam_search(start_token, max_len=10, beam_width=3, eos_token_id=eos_id)
    print("Beam Search 结果:", beam_res)
