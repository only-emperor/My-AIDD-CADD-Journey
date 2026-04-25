import pandas as pd
import numpy as np
import re, json, os, math, time, random
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error

# ─────────────────────────────────────────
# 全局配置
# ─────────────────────────────────────────
SEED       = 42
MAX_LEN    = 100
BATCH_SIZE = 64
EPOCHS     = 20
CSV_FILE   = '/content/SweetpredDB.csv'

C_BLUE = (75/255, 116/255, 178/255)
C_RED  = (219/255, 49/255,  36/255)

def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

seed_everything()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:   plt.style.use('seaborn-v0_8-whitegrid')
except: plt.style.use('ggplot')
plt.rcParams['axes.unicode_minus'] = False
# ─────────────────────────────────────────
# 1. SMILES Tokenizer
# ─────────────────────────────────────────
class SMILESTokenizer:
    PAT = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    SPECIAL = ['<pad>', '<unk>', '<sos>', '<eos>', '<mask>']

    def __init__(self, max_len=MAX_LEN, vocab=None):
        self.max_len = max_len
        self.regex   = re.compile(self.PAT)
        self.vocab   = vocab or {t: i for i, t in enumerate(self.SPECIAL)}
        self.inv_vocab = {i: t for t, i in self.vocab.items()}

    # ── 核心方法 ──────────────────────────
    def tokenize(self, smi):
        return self.regex.findall(smi)

    def build_vocab(self, smiles_list):
        counter = Counter(t for s in smiles_list for t in self.tokenize(s))
        for tok, _ in counter.most_common():
            if tok not in self.vocab:
                idx = len(self.vocab)
                self.vocab[tok] = idx
                self.inv_vocab[idx] = tok
        print(f"✅ 词表构建完成 | 大小: {len(self.vocab)} | Top: {counter.most_common(1)[0]}")

    def encode(self, smi):
        ids = [self.vocab.get(t, 1) for t in self.tokenize(smi)[:self.max_len - 2]]
        ids = [2] + ids + [3]                           # <sos> … <eos>
        ids += [0] * (self.max_len - len(ids))          # padding
        return ids

    def decode(self, ids):
        return ''.join(self.inv_vocab.get(i, '<unk>') for i in ids
                       if self.inv_vocab.get(i) not in self.SPECIAL)

    # ── 持久化 ────────────────────────────
    def save(self, path='vocab.json'):
        json.dump(self.vocab, open(path, 'w'), indent=2)
        print(f"💾 词表已保存: {path}")

    @classmethod
    def load(cls, path, max_len=MAX_LEN):
        return cls(max_len=max_len, vocab=json.load(open(path)))


# ─────────────────────────────────────────
# 2. Dataset & DataLoader
# ─────────────────────────────────────────
class SweetnessDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.smiles = df['Smiles'].values
        self.labels = df['logSw'].values
        self.tok    = tokenizer

    def __len__(self): return len(self.smiles)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.tok.encode(self.smiles[idx]), dtype=torch.long),
            'labels':    torch.tensor(self.labels[idx],                  dtype=torch.float),
        }


def prepare_data(csv_path=CSV_FILE):
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    df.rename(columns={'smiles': 'Smiles', 'logsw': 'logSw'}, inplace=True)
    df.dropna(subset=['Smiles', 'logSw'], inplace=True)

    tok = SMILESTokenizer()
    tok.build_vocab(df['Smiles'].tolist())

    ds  = SweetnessDataset(df, tok)
    n   = len(ds)
    tv, vv, tsv = int(.8*n), int(.1*n), n - int(.8*n) - int(.1*n)
    tr_ds, va_ds, te_ds = random_split(ds, [tv, vv, tsv],
                                       generator=torch.Generator().manual_seed(SEED))

    mk = lambda d, sh: DataLoader(d, batch_size=BATCH_SIZE, shuffle=sh)
    print(f"📊 Train {tv} | Val {vv} | Test {tsv}")
    return tok, mk(tr_ds, True), mk(va_ds, False), mk(te_ds, False), df
def visualize_data(df, tok):
    seq_lens    = [len(tok.tokenize(s)) for s in df['Smiles']]
    tok_counter = Counter(t for s in df['Smiles'] for t in tok.tokenize(s))
    top10       = tok_counter.most_common(10)

    fig = plt.figure(figsize=(15, 7), dpi=130)
    fig.suptitle('Sweetness Dataset Dashboard', fontsize=16, fontweight='bold', y=.97)
    gs  = gridspec.GridSpec(2, 2, width_ratios=[1.5, 1])

    # ① 序列长度分布
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(seq_lens, bins=40, kde=True, color=C_BLUE, edgecolor='w', alpha=.7, ax=ax1)
    ax1.axvline(np.mean(seq_lens), color=C_RED, ls='--', lw=2, label=f'Mean={np.mean(seq_lens):.1f}')
    ax1.axvline(MAX_LEN, color='grey', ls=':', lw=2, label=f'MaxLen={MAX_LEN}')
    ax1.set_title('Sequence Length Distribution', fontweight='bold', loc='left')
    ax1.legend()

    # ② Top-10 Token
    ax2 = fig.add_subplot(gs[:, 1])
    labels, counts = zip(*top10)
    bars = sns.barplot(x=list(counts), y=list(labels), ax=ax2, orient='h',
                       color=C_BLUE)
    bars.patches[0].set_facecolor(C_RED)
    ax2.set_title('Top 10 Tokens', fontweight='bold', loc='left')
    for i, v in enumerate(counts):
        ax2.text(v + 2, i, str(v), va='center', fontsize=9)

    # ③ logSw 分布
    ax3 = fig.add_subplot(gs[1, 0])
    sns.histplot(df['logSw'], bins=30, kde=True, color=C_BLUE, edgecolor='w', alpha=.6,
                 stat='density', ax=ax3)
    ax3.axvline(3.0, color=C_RED, ls='--', lw=2)
    ax3.text(3.05, ax3.get_ylim()[1]*.85, '>3 High', color=C_RED, fontweight='bold')
    ax3.set_title('Label Distribution (logSw)', fontweight='bold', loc='left')

    plt.tight_layout(); plt.show()
# ─────────────────────────────────────────
# Bi-LSTM
# ─────────────────────────────────────────
class BiLSTMRegressor(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64,
                 num_layers=2, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=.2 if num_layers > 1 else 0)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.ReLU(), nn.Dropout(.2), nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(self.embedding(x))   # [B, L, H*2]
        return self.regressor(out.mean(1)).squeeze(-1)


# ─────────────────────────────────────────
# Positional Encoding + Transformer
# ─────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(1e4) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerRegressor(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=3,
                 ffn_dim=256, pad_idx=0):
        super().__init__()
        self.pad_idx   = pad_idx
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc   = PositionalEncoding(d_model)
        enc_layer      = nn.TransformerEncoderLayer(d_model, nhead, ffn_dim,
                                                    dropout=.1, batch_first=True)
        self.encoder   = nn.TransformerEncoder(enc_layer, num_layers)
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(.1), nn.Linear(64, 1)
        )

    def forward(self, x):
        pad_mask = (x == self.pad_idx)                         # [B, L]
        emb      = self.pos_enc(self.embedding(x) * math.sqrt(x.size(-1)))
        enc      = self.encoder(emb, src_key_padding_mask=pad_mask)

        # masked mean pooling
        mask_f   = (~pad_mask).unsqueeze(-1).float()          # [B, L, 1]
        feat     = (enc * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-9)
        return self.regressor(feat).squeeze(-1)
def run_training(model, train_dl, val_dl, test_dl, *, lr, label):
    """通用训练循环，返回 history 和测试结果。"""
    model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    history = {'train_loss': [], 'val_loss': [], 'val_r2': []}

    print(f"\n🚀 训练 [{label}]  device={DEVICE}")
    t0 = time.time()

    for ep in range(EPOCHS):
        # ── train ──
        model.train()
        t_loss = sum(
            _step(model, batch, criterion, opt, train=True)
            for batch in train_dl
        ) / len(train_dl)

        # ── val ──
        v_loss, v_r2, *_ = _evaluate(model, val_dl, criterion)

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['val_r2'].append(v_r2)
        print(f"  Ep {ep+1:02d}/{EPOCHS} | train={t_loss:.4f} | val={v_loss:.4f} | R²={v_r2:.4f}")

    print(f"✅ 完成  ({time.time()-t0:.1f}s)")

    # ── test ──
    te_loss, te_r2, y_true, y_pred = _evaluate(model, test_dl, criterion)
    print(f"🧪 Test  MSE={te_loss:.4f}  R²={te_r2:.4f}")

    torch.save(model.state_dict(), f"{label.lower().replace(' ','_')}.pt")
    return history, y_true, y_pred


# ── 辅助函数 ──────────────────────────────
def _step(model, batch, criterion, opt, train=True):
    ids, lbl = batch['input_ids'].to(DEVICE), batch['labels'].to(DEVICE)
    pred = model(ids)
    loss = criterion(pred, lbl)
    if train:
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return loss.item()

def _evaluate(model, loader, criterion):
    model.eval()
    preds, labels, total = [], [], 0
    with torch.no_grad():
        for batch in loader:
            ids, lbl = batch['input_ids'].to(DEVICE), batch['labels'].to(DEVICE)
            pred = model(ids)
            total += criterion(pred, lbl).item()
            preds.extend(pred.cpu().numpy())
            labels.extend(lbl.cpu().numpy())
    y_true, y_pred = np.array(labels), np.array(preds)
    return total / len(loader), r2_score(y_true, y_pred), y_true, y_pred
def plot_results(label, color, history, y_true, y_pred):
    """单模型：Loss曲线 + 散点图"""
    ep = range(1, EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=110)

    ax1.plot(ep, history['train_loss'], '--', color=color, label='Train')
    ax1.plot(ep, history['val_loss'],   '-',  color=color, label='Val',  lw=2)
    ax1.set(title=f'{label} Learning Curve', xlabel='Epoch', ylabel='MSE')
    ax1.legend(); ax1.grid(alpha=.3)

    ax2.scatter(y_true, y_pred, alpha=.5, color=color, s=18)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax2.plot([lo, hi], [lo, hi], 'k--')
    ax2.set(title=f'{label}  R²={r2_score(y_true, y_pred):.3f}',
            xlabel='True', ylabel='Predicted')
    ax2.grid(alpha=.3)

    plt.tight_layout(); plt.show()


def plot_showdown(lstm_h, trans_h):
    """双模型对比：Val Loss & Val R²"""
    ep = range(1, EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=120)

    # ── Loss ──
    for h, c, m, lab in [(lstm_h, C_BLUE, 'o', 'Bi-LSTM'),
                          (trans_h, C_RED,  's', 'Transformer')]:
        ax1.plot(ep, h['val_loss'], f'{m}-', color=c, label=lab, lw=2, ms=5)
    ax1.fill_between(ep,
                     lstm_h['val_loss'], trans_h['val_loss'],
                     where=np.array(trans_h['val_loss']) < np.array(lstm_h['val_loss']),
                     color=C_RED, alpha=.12, label='Trans wins')
    ax1.fill_between(ep,
                     lstm_h['val_loss'], trans_h['val_loss'],
                     where=np.array(trans_h['val_loss']) >= np.array(lstm_h['val_loss']),
                     color=C_BLUE, alpha=.12, label='LSTM wins')
    ax1.set(title='Val Loss (↓ better)', xlabel='Epoch', ylabel='MSE')
    ax1.legend(); ax1.grid(alpha=.3)

    # ── R² ──
    for h, c, m, lab in [(lstm_h, C_BLUE, 'o', 'Bi-LSTM'),
                          (trans_h, C_RED,  's', 'Transformer')]:
        ax2.plot(ep, h['val_r2'], f'{m}-', color=c, label=lab, lw=2, ms=5)
        best_idx = int(np.argmax(h['val_r2']))
        ax2.annotate(f"{lab}: {h['val_r2'][best_idx]:.3f}",
                     xy=(best_idx + 1, h['val_r2'][best_idx]),
                     xytext=(0, 12), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color=c), color=c, fontsize=9)
    ax2.set(title='Val R² (↑ better)', xlabel='Epoch', ylabel='R²')
    ax2.legend(loc='lower right'); ax2.grid(alpha=.3)

    plt.tight_layout(); plt.show()


def print_report(lstm_h, trans_h):
    bl, bt = max(lstm_h['val_r2']), max(trans_h['val_r2'])
    winner = "Transformer ✅" if bt - bl > .01 else ("Bi-LSTM 🔵" if bl - bt > .01 else "平局 🤝")
    print(f"\n{'='*50}")
    print(f"  Bi-LSTM    best R²: {bl:.4f}")
    print(f"  Transformer best R²: {bt:.4f}")
    print(f"  结论 → {winner}")
    print(f"{'='*50}\n")
# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
if __name__ == '__main__':

    # 1. 数据
    tok, train_dl, val_dl, test_dl, raw_df = prepare_data()
    tok.save()

    # 2. 数据可视化
    visualize_data(raw_df, tok)

    pad_id    = tok.vocab['<pad>']
    vsize     = len(tok.vocab)

    # 3. Bi-LSTM
    lstm_model   = BiLSTMRegressor(vsize, pad_idx=pad_id)
    lstm_h, lstm_yt, lstm_yp = run_training(
        lstm_model, train_dl, val_dl, test_dl, lr=1e-3, label='BiLSTM')
    plot_results('Bi-LSTM', C_BLUE, lstm_h, lstm_yt, lstm_yp)

    # 4. Transformer
    trans_model  = TransformerRegressor(vsize, pad_idx=pad_id)
    trans_h, trans_yt, trans_yp = run_training(
        trans_model, train_dl, val_dl, test_dl, lr=5e-4, label='Transformer')
    plot_results('Transformer', C_RED, trans_h, trans_yt, trans_yp)

    # 5. 对比
    plot_showdown(lstm_h, trans_h)
    print_report(lstm_h, trans_h)
