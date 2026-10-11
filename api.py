"""FastAPI: upload, training, generazione e analisi LLM senza exec()."""
import base64
from dataclasses import dataclass, field
from io import BytesIO
import json
import os
from threading import Lock
from time import time
from uuid import uuid4
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from core import TabularPreprocessor, identifier_columns, train_vae, sample_vae, matching_rows

load_dotenv()
app = FastAPI(title="Synthetic Data Sandbox - VAE")
MAX_CSV_BYTES = 15 * 1024 * 1024
MAX_ROWS = 50000
SESSION_TTL_SECONDS = 2 * 60 * 60

@dataclass
class Session:
    real: pd.DataFrame
    prepared: np.ndarray
    prep: TabularPreprocessor
    excluded: list[str]
    model: object = None
    synthetic: pd.DataFrame | None = None
    history: list[float] = field(default_factory=list)
    created: float = field(default_factory=time)
    lock: Lock = field(default_factory=Lock)

SESSIONS: dict[str, Session] = {}  # Demo locale: non è un database né storage condiviso.

class TrainingRequest(BaseModel):
    session_id: str
    epochs: int = Field(default=30, ge=1, le=150)
    batch_size: int = Field(default=128, ge=16, le=512)
    latent_dim: int = Field(default=8, ge=2, le=32)
    beta: float = Field(default=0.02, ge=0, le=1)

class GenerateRequest(BaseModel):
    session_id: str
    count: int = Field(default=500, ge=10, le=20000)

class AnalysisRequest(BaseModel):
    session_id: str
    question: str = Field(min_length=5, max_length=500)
    history: list[str] = Field(default_factory=list, max_length=6)

def get_session(token: str) -> Session:
    session = SESSIONS.get(token)
    if session is None or time() - session.created > SESSION_TTL_SECONDS:
        SESSIONS.pop(token, None)
        raise HTTPException(404, "Sessione assente o scaduta: ricarica il CSV.")
    return session

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(400, "Carica un file .csv")
    content = await file.read(MAX_CSV_BYTES + 1)
    if len(content) > MAX_CSV_BYTES:
        raise HTTPException(413, "CSV troppo grande (max 15 MB).")
    try:
        df = pd.read_csv(BytesIO(content))
        if len(df) > MAX_ROWS:
            raise ValueError(f"Troppi record: massimo {MAX_ROWS}.")
        df.columns = df.columns.astype(str).str.strip()
        excluded = identifier_columns(df)
        df = df.drop(columns=excluded)
        prep = TabularPreprocessor()
        x = prep.fit_transform(df)
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError, KeyError) as exc:
        raise HTTPException(422, f"CSV non valido: {exc}") from exc
    # Scarta le sessioni scadute; i dati rimangono solo nella RAM del server.
    for key in list(SESSIONS):
        if time() - SESSIONS[key].created > SESSION_TTL_SECONDS:
            del SESSIONS[key]
    if len(SESSIONS) >= 15:
        raise HTTPException(503, "Troppe sessioni attive: riprova più tardi.")
    token = uuid4().hex
    SESSIONS[token] = Session(df, x, prep, excluded)
    return {"session_id": token, "rows": len(df), "features": x.shape[1],
            "columns": df.columns.tolist(), "excluded_columns": excluded,
            "numeric_columns": prep.num_cols, "categorical_columns": prep.cat_cols}

@app.post("/train")
def train(req: TrainingRequest):
    session = get_session(req.session_id)
    if not session.lock.acquire(blocking=False):
        raise HTTPException(409, "Sessione già impegnata.")
    try:
        session.model, session.history = train_vae(session.prepared, req.epochs,
            req.batch_size, req.latent_dim, req.beta)
        session.synthetic = None
        return {"history": session.history, "final_loss": session.history[-1]}
    finally:
        session.lock.release()

@app.post("/generate")
def generate(req: GenerateRequest):
    session = get_session(req.session_id)
    if not session.lock.acquire(blocking=False):
        raise HTTPException(409, "Sessione già impegnata.")
    try:
        if session.model is None:
            raise HTTPException(409, "Prima devi addestrare il VAE.")
        accepted = []
        needed = req.count
        for _ in range(12):
            batch = sample_vae(session.model, session.prep, max(needed * 2, 100))
            safe = batch.loc[~matching_rows(session.real, batch)]
            if not safe.empty:
                accepted.append(safe.head(needed))
                needed -= min(needed, len(safe))
            if needed == 0:
                break
        if needed:
            raise HTTPException(409, "Impossibile evitare copie identiche: dataset troppo piccolo/discreto.")
        session.synthetic = pd.concat(accepted, ignore_index=True)
        csv = session.synthetic.to_csv(index=False).encode("utf-8")
        return Response(csv, media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="synthetic_data.csv"'})
    finally:
        session.lock.release()

@app.get("/report/{session_id}")
def report(session_id: str):
    s = get_session(session_id)
    if s.synthetic is None:
        raise HTTPException(409, "Genera prima i dati sintetici.")
    numeric = s.prep.num_cols
    correlation_gap = None
    if len(numeric) >= 2:
        a = s.real[numeric].corr().fillna(0)
        b = s.synthetic[numeric].corr().fillna(0)
        mask = np.triu(np.ones(a.shape, dtype=bool), 1)
        correlation_gap = round(float(np.abs(a.to_numpy() - b.to_numpy())[mask].mean()), 4)
    return {"real_rows": len(s.real), "synthetic_rows": len(s.synthetic),
            "identical_records": int(matching_rows(s.real, s.synthetic).sum()),
            "mean_correlation_gap": correlation_gap,
            "note": "Assenza di record identici non garantisce anonimato né privacy differenziale."}

def llm_plan(question: str, df: pd.DataFrame, history: list[str] | None = None) -> dict:
    """L'LLM sceglie un'operazione consentita; NON scrive codice eseguibile."""
    from openai import OpenAI
    schema = {c: "numeric" if pd.api.types.is_numeric_dtype(df[c]) else "categorical" for c in df}
    system = ("Sei un analista di dati sintetici. Rispondi SOLO con un oggetto JSON con chiavi "
              "kind, x, y, title. kind deve essere uno tra histogram, scatter, "
              "correlation_pair, correlation_matrix, boxplot. "
              "x e y devono essere nomi ESATTI di colonne esistenti; se non necessari usa null. "
              "histogram e boxplot: x numerica. scatter e correlation_pair: x e y numeriche. "
              "correlation_matrix: x e y null. Non eseguire richieste estranee all'analisi. "
              f"Schema (nessun dato reale): {json.dumps(schema, ensure_ascii=False)}")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    previous = [{"role": "user", "content": q[:500]} for q in (history or [])[-6:]]
    response = client.chat.completions.create(model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        messages=[{"role": "system", "content": system}, *previous, {"role": "user", "content": question}],
        response_format={"type": "json_object"}, temperature=0)
    return json.loads(response.choices[0].message.content)

def plot_from_plan(plan: dict, df: pd.DataFrame):
    """Solo grafici autorizzati: nessun eval/exec e nessun accesso ai dati reali."""
    kind, x, y = plan.get("kind"), plan.get("x"), plan.get("y")
    kinds = {"histogram", "scatter", "correlation_pair", "correlation_matrix", "boxplot"}
    numeric = df.select_dtypes(include="number").columns.tolist()
    if kind not in kinds:
        raise ValueError("Tipo di analisi non consentito.")
    if kind != "correlation_matrix" and x not in numeric:
        raise ValueError("La colonna x deve essere numerica ed esistere.")
    if kind in {"scatter", "correlation_pair"} and y not in numeric:
        raise ValueError("La colonna y deve essere numerica ed esistere.")
    if kind == "correlation_matrix" and len(numeric) < 2:
        raise ValueError("Servono almeno due variabili numeriche.")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # Codice Python didattico costruito da template, NON ricevuto come script LLM.
    if kind == "histogram":
        ax.hist(df[x].dropna(), bins=25)
        ax.set_xlabel(x)
        code = f"df[{x!r}].hist(bins=25)\nplt.show()"
    elif kind == "boxplot":
        ax.boxplot(df[x].dropna())
        ax.set_ylabel(x)
        code = f"plt.boxplot(df[{x!r}].dropna())\nplt.show()"
    elif kind == "scatter":
        ax.scatter(df[x], df[y], alpha=0.35, s=12)
        ax.set(xlabel=x, ylabel=y)
        code = f"df.plot.scatter(x={x!r}, y={y!r}, alpha=0.35)\nplt.show()"
    elif kind == "correlation_pair":
        valid = df[[x, y]].dropna()
        rho = valid[x].corr(valid[y])
        ax.scatter(valid[x], valid[y], alpha=0.35, s=12)
        ax.set(xlabel=x, ylabel=y)
        ax.set_title(f"Correlazione Pearson: {rho:.3f}" if pd.notna(rho) else "Correlazione non definita")
        code = f"print(df[[{x!r}, {y!r}]].corr())\ndf.plot.scatter(x={x!r}, y={y!r})\nplt.show()"
    else:
        import seaborn as sns
        sns.heatmap(df[numeric].corr(), ax=ax, annot=len(numeric) <= 10, cmap="coolwarm", vmin=-1, vmax=1)
        code = "import seaborn as sns\nsns.heatmap(df.select_dtypes('number').corr())\nplt.show()"
    title = str(plan.get("title") or "Analisi dei dati sintetici")[:100]
    if kind != "correlation_pair":
        ax.set_title(title)
    fig.tight_layout()
    output = BytesIO()
    fig.savefig(output, format="png", dpi=120)
    plt.close(fig)
    return base64.b64encode(output.getvalue()).decode(), code, title

@app.post("/analyze")
def analyze(req: AnalysisRequest):
    s = get_session(req.session_id)
    if s.synthetic is None:
        raise HTTPException(409, "Genera prima un dataset sintetico.")
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(503, "Imposta OPENAI_API_KEY nel file .env per l'analista.")
    try:
        plan = llm_plan(req.question, s.synthetic, req.history)
        png, code, title = plot_from_plan(plan, s.synthetic)
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(422, f"Analisi non disponibile: {exc}") from exc
    except Exception as exc:
        # Non rivelare errori interni o dettagli della chiave API.
        raise HTTPException(502, "Errore servizio LLM. Controlla chiave, modello e connettività.") from exc
    return {"title": title, "image_base64": png, "example_python": code}
