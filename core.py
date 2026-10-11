"""Preprocessing, VAE tabellare e generazione di record sintetici."""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

# Identificatori diretti: non devono entrare nel modello generativo.
ID_NAMES = {"id", "name", "nome", "surname", "cognome", "email", "phone", "telefono",
            "address", "indirizzo", "ssn", "iban", "codice_fiscale", "tax_id"}

def identifier_columns(df: pd.DataFrame) -> list[str]:
    """Individua alcune colonne anagrafiche comuni (controllo non esaustivo)."""
    return [c for c in df if c.lower().strip().replace(" ", "_") in ID_NAMES
            or c.lower().strip().replace(" ", "_").endswith("_id")]

@dataclass
class TabularPreprocessor:
    num_cols: list[str] = field(default_factory=list)
    cat_cols: list[str] = field(default_factory=list)
    int_cols: list[str] = field(default_factory=list)
    num_fill: dict = field(default_factory=dict)
    cat_fill: dict = field(default_factory=dict)
    scaler: MinMaxScaler | None = None
    encoder: OneHotEncoder | None = None
    cat_sizes: list[int] = field(default_factory=list)
    feature_count: int = 0

    def _numeric(self, df: pd.DataFrame) -> np.ndarray:
        if not self.num_cols:
            return np.empty((len(df), 0), dtype=np.float32)
        part = df[self.num_cols].apply(pd.to_numeric, errors="coerce").fillna(self.num_fill)
        return self.scaler.transform(part).astype(np.float32)

    def _categorical(self, df: pd.DataFrame) -> np.ndarray:
        if not self.cat_cols:
            return np.empty((len(df), 0), dtype=np.float32)
        part = df[self.cat_cols].astype("string").fillna(self.cat_fill).astype(str)
        return self.encoder.transform(part).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Imputa valori mancanti, applica MinMaxScaler e One-Hot Encoding."""
        if df.empty or len(df) < 20:
            raise ValueError("Servono almeno 20 righe e almeno una colonna.")
        if df.columns.duplicated().any():
            raise ValueError("Il CSV contiene nomi di colonne duplicati.")
        if len(df.columns) > 100:
            raise ValueError("Massimo 100 colonne per questa demo.")
        self.num_cols = df.select_dtypes(include="number").columns.tolist()
        self.cat_cols = [c for c in df if c not in self.num_cols]
        self.int_cols = [c for c in self.num_cols if pd.api.types.is_integer_dtype(df[c])]
        parts = []
        if self.num_cols:
            numeric = df[self.num_cols].apply(pd.to_numeric, errors="coerce")
            self.num_fill = numeric.median().fillna(0).to_dict()
            self.scaler = MinMaxScaler()
            self.scaler.fit(numeric.fillna(self.num_fill))
            parts.append(self._numeric(df))
        if self.cat_cols:
            self.cat_fill = {}
            for c in self.cat_cols:
                values = df[c].astype("string").dropna()
                self.cat_fill[c] = str(values.mode().iloc[0]) if not values.empty else "MISSING"
            cat = df[self.cat_cols].astype("string").fillna(self.cat_fill).astype(str)
            if any(cat[c].nunique() > 50 for c in self.cat_cols):
                raise ValueError("Una colonna categorica supera 50 valori distinti: rimuovere ID/testo libero.")
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            encoded = self.encoder.fit_transform(cat).astype(np.float32)
            self.cat_sizes = [len(x) for x in self.encoder.categories_]
            parts.append(encoded)
        self.feature_count = sum(p.shape[1] for p in parts)
        if self.feature_count > 512:
            raise ValueError("Troppi attributi dopo il One-Hot Encoding (max 512).")
        return np.concatenate(parts, axis=1)

    def inverse_transform(self, array: np.ndarray) -> pd.DataFrame:
        """Riporta i vettori del decoder in colonne comprensibili."""
        result = {}
        offset = len(self.num_cols)
        if self.num_cols:
            values = self.scaler.inverse_transform(np.clip(array[:, :offset], 0, 1))
            for i, c in enumerate(self.num_cols):
                result[c] = np.rint(values[:, i]).astype(np.int64) if c in self.int_cols else values[:, i]
        if self.cat_cols:
            for c, n, categories in zip(self.cat_cols, self.cat_sizes, self.encoder.categories_):
                block = array[:, offset:offset + n]
                result[c] = categories[np.argmax(block, axis=1)]
                offset += n
        return pd.DataFrame(result)[self.num_cols + self.cat_cols]

class TabularVAE(nn.Module):
    """VAE: encoder -> media/log-varianza -> campionamento -> decoder."""
    def __init__(self, input_dim: int, latent_dim: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, 64), nn.ReLU())
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(),
                                     nn.Linear(64, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, input_dim), nn.Sigmoid())

    def forward(self, x: torch.Tensor):
        features = self.encoder(x)
        mu, logvar = self.mu(features), self.logvar(features)
        eps = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * logvar) * eps  # Reparameterization trick.
        return self.decoder(z), mu, logvar

def train_vae(x: np.ndarray, epochs: int, batch_size: int, latent_dim: int, beta: float):
    """MSE di ricostruzione + beta * KL divergence verso N(0,1)."""
    torch.manual_seed(42)
    model = TabularVAE(x.shape[1], latent_dim)
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=True)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    model.train()
    for _ in range(epochs):
        total = 0.0
        for (batch,) in loader:
            recon, mu, logvar = model(batch)
            mse = F.mse_loss(recon, batch)
            kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.square() - logvar.exp(), dim=1))
            loss = mse + beta * kl / latent_dim
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item() * len(batch)
        history.append(round(total / len(x), 6))
    model.eval()
    return model, history

@torch.no_grad()
def sample_vae(model: TabularVAE, preprocessor: TabularPreprocessor, count: int) -> pd.DataFrame:
    """Genera dati con z ~ N(0,1), senza utilizzare l'encoder."""
    pieces = []
    for n in range(0, count, 1024):
        z = torch.randn(min(1024, count - n), model.latent_dim)
        pieces.append(model.decoder(z).numpy())
    return preprocessor.inverse_transform(np.concatenate(pieces))

def matching_rows(real: pd.DataFrame, synthetic: pd.DataFrame) -> np.ndarray:
    """Maschera dei record perfettamente identici a una riga del dataset reale."""
    # Confronto esatto sull'intera riga, senza copiare alcuna riga originale.
    columns = synthetic.columns.tolist()
    originals = pd.MultiIndex.from_frame(real[columns].drop_duplicates())
    generated = pd.MultiIndex.from_frame(synthetic[columns])
    return np.asarray(generated.isin(originals), dtype=bool)
