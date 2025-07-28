import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from .config import MATCH_DATA_PATH


# ─────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────
class Dota2DraftDataset(Dataset):
    def __init__(self, data: pd.DataFrame, n_heroes: int):
        self.data = data.reset_index(drop=True)
        self.n_heroes = n_heroes

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        m = self.data.iloc[idx]

        radiant = list(map(int, m["radiant_draft"]))
        dire    = list(map(int, m["dire_draft"]))
        if len(radiant) != 5 or len(dire) != 5:
            raise ValueError(f"row {idx}: draft length not 5+5")

        draft = radiant + dire
        bad   = [h for h in draft if h >= self.n_heroes or h < 0]
        if bad:
            raise IndexError(
                f"row {idx}: hero IDs {bad} exceed allowed range 0‑{self.n_heroes-1}"
            )

        draft_tensor  = torch.tensor(draft, dtype=torch.long)
        label_tensor  = torch.tensor(m["radiant_wins"], dtype=torch.float32)  # ← no unsqueeze
        return draft_tensor, label_tensor


# ─────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────
class DraftTransformer(nn.Module):
    def __init__(self, n_heroes: int, d_model: int = 32,
                 n_heads: int = 4, n_layers: int = 2, d_ff: int = 64):
        super().__init__()
        self.seq_len   = 10
        self.hero_emb  = nn.Embedding(n_heroes, d_model)
        self.pos_emb   = nn.Embedding(self.seq_len, d_model)
        enc_layer      = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout=0.1, batch_first=True)
        self.encoder   = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head      = nn.Linear(d_model, 1)
        self.register_buffer("pos_idx", torch.arange(self.seq_len), persistent=False)

    def forward(self, draft: torch.Tensor) -> torch.Tensor:   # draft (B,10)
        h = self.hero_emb(draft) + self.pos_emb(self.pos_idx)
        h = self.encoder(h)
        return self.head(h[:, 0])   # (B,1)


# ─────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────
def train_step(model, draft, tgt, crit, opt):
    model.train()
    opt.zero_grad()
    logits = model(draft).squeeze(1)        # (B)
    loss   = crit(logits, tgt)              # tgt (B)
    loss.backward()
    opt.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, draft, tgt, crit):
    model.eval()
    logits = model(draft).squeeze(1)
    return crit(logits, tgt).item()


def train_model(model, train_loader, val_loader, crit, opt, epochs: int = 5):
    for ep in range(epochs):
        loss_sum = 0.0
        for draft, tgt in train_loader:
            loss_sum += train_step(model, draft, tgt, crit, opt)
        print(f"Epoch {ep+1}/{epochs} | train loss {loss_sum/len(train_loader):.4f}")

        v_draft, v_tgt = next(iter(val_loader))
        v_loss = evaluate(model, v_draft, v_tgt, crit)
        print(f"                 | valid loss {v_loss:.4f}")


# ─────────────────────────────────────────────────────────────────────
def main():
    try:
        df = pd.read_parquet(MATCH_DATA_PATH)
        print(f"Loaded {len(df):,} rows from {MATCH_DATA_PATH}")
    except Exception as e:
        print(f"Load failed: {e}")
        return

    # vocabulary size
    all_ids = pd.concat(
        [df["radiant_draft"].explode(), df["dire_draft"].explode()]
    ).astype(int)
    n_heroes = int(all_ids.max()) + 1
    print(f"embedding size n_heroes = {n_heroes}")

    # split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # loaders
    batch = 32
    train_loader = DataLoader(Dota2DraftDataset(train_df, n_heroes),
                              batch_size=batch, shuffle=True)
    val_loader   = DataLoader(Dota2DraftDataset(val_df, n_heroes),
                              batch_size=batch, shuffle=False)

    # model + optimiser
    model     = DraftTransformer(n_heroes)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10)


if __name__ == "__main__":
    main()
