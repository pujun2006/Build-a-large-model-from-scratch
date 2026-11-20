import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# 工程上设置随机种子
def set_seed(seed=42):
    random.seed(seed)  # Python 原生随机库
    np.random.seed(seed)  # NumPy 随机库
    torch.manual_seed(seed)  # PyTorch CPU 随机种子
    torch.cuda.manual_seed(seed)  # PyTorch 当前 GPU 随机种子
    torch.cuda.manual_seed_all(seed)  # PyTorch 所有 GPU 随机种子（多卡训练）

    # 保证 CUDA 底层卷积算法的选择也是确定性的（可选，可能略微影响速度）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 在主程序一开始调用
set_seed(42)


# 定义编码器
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))  # (batch_size, src_len, emb_dim)
        outputs, (hidden, cell) = self.rnn(embedded)  # outputs: (batch_size, src_len, hidden_dim)
        return hidden, cell


# 定义解码器
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, hidden, cell):
        trg = trg.unsqueeze(1)  # 增加时间步维度 (batch_size, 1)
        embedded = self.dropout(self.embedding(trg))  # (batch_size, 1, emb_dim)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))  # output: (batch_size, 1, hidden_dim)
        prediction = self.fc_out(output.squeeze(1))  # (batch_size, output_dim)
        return prediction, hidden, cell


# 定义Seq2Seq模型
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # 编码器的输出作为解码器的初始隐藏状态
        hidden, cell = self.encoder(src)
        input = trg[:, 0]  # 解码器的第一个输入是<sos>标记

        # 逐步解码目标序列
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)  # 获取预测的最高分值
            input = trg[:, t] if teacher_force else top1

        return outputs


# 超参数设置
INPUT_DIM = 10  # 输入词典大小
OUTPUT_DIM = 10  # 输出词典大小
ENC_EMB_DIM = 16  # 编码器嵌入维度
DEC_EMB_DIM = 16  # 解码器嵌入维度
HIDDEN_DIM = 32  # 隐藏层维度
N_LAYERS = 2  # 编码器和解码器层数
ENC_DROPOUT = 0.5  # 编码器dropout
DEC_DROPOUT = 0.5  # 解码器dropout
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 实例化模型
encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HIDDEN_DIM, N_LAYERS, ENC_DROPOUT)
decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HIDDEN_DIM, N_LAYERS, DEC_DROPOUT)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

# 损失函数和优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


# 训练模型
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for src, trg in iterator:
        src = src.to(DEVICE)
        trg = trg.to(DEVICE)

        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


# 模拟数据生成器
def generate_dummy_data(batch_size, seq_len, vocab_size):
    src = torch.randint(1, vocab_size, (batch_size, seq_len))
    trg = torch.randint(1, vocab_size, (batch_size, seq_len))
    return src, trg


# 模拟训练
BATCH_SIZE = 32
SEQ_LEN = 5
VOCAB_SIZE = 10
N_EPOCHS = 5
CLIP = 1

for epoch in range(N_EPOCHS):
    src, trg = generate_dummy_data(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    iterator = [(src, trg)]
    train_loss = train(model, iterator, optimizer, criterion, CLIP)
    print(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}')
