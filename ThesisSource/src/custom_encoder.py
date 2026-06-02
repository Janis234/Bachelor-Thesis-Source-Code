import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


EMB_DIM = 32
HIDDEN_DIM = 64
DROPOUT = 0.3
DEFAULT_EPOCHS = 15
DEFAULT_BATCH = 1024
DEFAULT_LR = 1e-3
DEFAULT_VAL_SPLIT = 0.2
SEED = 42


class GenreClassifier(nn.Module):

    def __init__(self, n_features: int, n_genres: int,
                 hidden_dim: int = HIDDEN_DIM, emb_dim: int = EMB_DIM):
        super().__init__()
        self.feature_to_hidden = nn.Linear(n_features, hidden_dim)
        self.hidden_to_emb = nn.Linear(hidden_dim, emb_dim)
        self.emb_to_genre = nn.Linear(emb_dim, n_genres)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(DROPOUT)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.feature_to_hidden(x))
        h = self.drop(h)
        return self.act(self.hidden_to_emb(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb_to_genre(self.embed(x))


def train_encoder(
    audio_matrix: np.ndarray,
    genre_labels: pd.Series,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH,
    lr: float = DEFAULT_LR,
    val_split: float = DEFAULT_VAL_SPLIT,
    seed: int = SEED,
    verbose: bool = True,
) -> tuple:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    genre_categories = pd.Categorical(genre_labels)
    n_genres = len(genre_categories.categories)
    y = genre_categories.codes.astype(np.int64)

    n = len(audio_matrix)
    idx = rng.permutation(n)
    n_val = int(n * val_split)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train = torch.from_numpy(audio_matrix[train_idx]).float()
    y_train = torch.from_numpy(y[train_idx])
    X_val = torch.from_numpy(audio_matrix[val_idx]).float()
    y_val = torch.from_numpy(y[val_idx])

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    model = GenreClassifier(n_features=audio_matrix.shape[1], n_genres=n_genres)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimiser.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(X_train)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == y_val).float().mean().item()

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_acc": val_acc,
        })
        if verbose:
            print(f"Epoch {epoch + 1:>2}/{epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    info = {
        "n_genres": n_genres,
        "genre_categories": list(genre_categories.categories),
        "history": history,
        "final_val_acc": history[-1]["val_acc"],
    }
    return model, info


def extract_embeddings(model: GenreClassifier, audio_matrix: np.ndarray,
                       batch_size: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    X = torch.from_numpy(audio_matrix).float()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            out.append(model.embed(X[i:i + batch_size]).numpy())
    return np.vstack(out)


def encode_target_features(model: GenreClassifier, target_vec: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(target_vec).float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return model.embed(x).numpy()


def save_encoder(model: GenreClassifier, info: dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "n_genres": info["n_genres"],
        "genre_categories": info["genre_categories"],
        "n_features": model.feature_to_hidden.in_features,
        "hidden_dim": model.feature_to_hidden.out_features,
        "emb_dim": model.hidden_to_emb.out_features,
    }
    torch.save(payload, path)


def load_encoder(path: str) -> tuple:
    payload = torch.load(path, weights_only=False)
    model = GenreClassifier(
        n_features=payload["n_features"],
        n_genres=payload["n_genres"],
        hidden_dim=payload["hidden_dim"],
        emb_dim=payload["emb_dim"],
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    info = {
        "n_genres": payload["n_genres"],
        "genre_categories": payload["genre_categories"],
    }
    return model, info


def train_or_load(
    df: pd.DataFrame,
    audio_matrix: np.ndarray,
    model_path: str = "models/custom_encoder.pt",
    emb_path: str = "data/custom_audio_embeddings.npy",
    force_retrain: bool = False,
    **train_kwargs,
) -> tuple:
    if not force_retrain and os.path.exists(model_path) and os.path.exists(emb_path):
        model, info = load_encoder(model_path)
        embeddings = np.load(emb_path)
        return model, info, embeddings

    model, info = train_encoder(audio_matrix, df["genre"], **train_kwargs)
    embeddings = extract_embeddings(model, audio_matrix)
    save_encoder(model, info, model_path)
    os.makedirs(os.path.dirname(emb_path) or ".", exist_ok=True)
    np.save(emb_path, embeddings)
    return model, info, embeddings
