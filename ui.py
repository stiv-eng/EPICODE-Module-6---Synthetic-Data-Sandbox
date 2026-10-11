"""Dashboard Streamlit per il laboratorio Synthetic Data Sandbox."""
import base64
from io import BytesIO
import os
import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
st.set_page_config(page_title="Synthetic Data Sandbox", layout="wide")
st.title("Synthetic Data Sandbox | VAE Edition")
st.caption("Laboratorio locale: i dati sensibili richiedono ulteriori verifiche di privacy e sicurezza.")

def post(path: str, **kwargs):
    """Invia una richiesta a FastAPI e mostra errori leggibili."""
    try:
        resp = requests.post(API_URL + path, timeout=300, **kwargs)
        if not resp.ok:
            try:
                detail = resp.json().get("detail", resp.text)
            except ValueError:
                detail = resp.text
            raise ValueError(str(detail))
        return resp
    except requests.exceptions.RequestException as exc:
        raise ValueError(f"Backend non raggiungibile: {exc}") from exc

def heatmap(df: pd.DataFrame, title: str):
    """Visualizza la matrice di correlazione numerica."""
    nums = df.select_dtypes(include="number")
    if nums.shape[1] < 2:
        st.info("Servono due variabili numeriche per la heatmap.")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(nums.corr(), ax=ax, cmap="coolwarm", vmin=-1, vmax=1,
                annot=nums.shape[1] <= 8)
    ax.set_title(title)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Lo stato permette a Streamlit di conservare upload e risultati tra i rerun.
for key, default in {"session_id": None, "real": None, "synthetic": None,
                     "training_history": None, "metadata": None, "chat": []}.items():
    if key not in st.session_state:
        st.session_state[key] = default

tab_data, tab_check, tab_llm = st.tabs(["1. Dati e VAE", "2. Validazione", "3. Analista LLM"])
with tab_data:
    uploaded = st.file_uploader("Carica un CSV tabellare", type="csv")
    if uploaded is not None and st.button("Prepara dataset"):
        try:
            with st.spinner("Preprocessing sul backend..."):
                info = post("/upload", files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")}).json()
            real = pd.read_csv(BytesIO(uploaded.getvalue())).drop(columns=info["excluded_columns"])
            st.session_state.update(session_id=info["session_id"], real=real,
                                    synthetic=None, training_history=None, metadata=info, chat=[])
            st.success(f"Pronte {info['rows']} righe, {info['features']} feature codificate.")
        except (ValueError, pd.errors.ParserError) as exc:
            st.error(str(exc))
    if st.session_state.session_id:
        info = st.session_state.metadata
        if info["excluded_columns"]:
            st.warning("Colonne escluse perché possibili identificatori: " + ", ".join(info["excluded_columns"]))
        st.subheader("Anteprima dei dati originali (colonne utilizzate)")
        st.dataframe(st.session_state.real.head(12), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            epochs = st.slider("Epoche", 5, 150, 30, 5)
        with c2:
            latent_dim = st.slider("Dimensione latente", 2, 32, 8)
        with c3:
            beta = st.slider("Peso KL (beta)", 0.0, 0.20, 0.02, 0.01)
        if st.button("Addestra VAE", type="primary"):
            try:
                with st.spinner("Training in esecuzione..."):
                    result = post("/train", json={"session_id": st.session_state.session_id,
                        "epochs": epochs, "latent_dim": latent_dim, "beta": beta}).json()
                st.session_state.training_history = result["history"]
                st.session_state.synthetic = None
                st.session_state.chat = []
                st.success(f"Training concluso. Loss finale: {result['final_loss']:.4f}")
            except ValueError as exc:
                st.error(str(exc))
        if st.session_state.training_history:
            st.line_chart(pd.DataFrame({"Loss": st.session_state.training_history}))
            amount = st.number_input("Righe sintetiche da produrre", min_value=10,
                                     max_value=20000, value=500, step=100)
            if st.button("Genera dataset sintetico"):
                try:
                    with st.spinner("Campionamento dal decoder..."):
                        response = post("/generate", json={"session_id": st.session_state.session_id,
                                                          "count": int(amount)})
                    st.session_state.synthetic = pd.read_csv(BytesIO(response.content))
                    st.session_state.chat = []
                    st.success("Dataset sintetico generato.")
                except (ValueError, pd.errors.ParserError) as exc:
                    st.error(str(exc))
        if st.session_state.synthetic is not None:
            synthetic = st.session_state.synthetic
            st.subheader("Anteprima sintetica")
            st.dataframe(synthetic.head(12), use_container_width=True)
            st.download_button("Scarica CSV sintetico", synthetic.to_csv(index=False).encode(),
                               file_name="synthetic_data.csv", mime="text/csv")

with tab_check:
    real, synthetic = st.session_state.real, st.session_state.synthetic
    if real is None or synthetic is None:
        st.info("Carica, addestra e genera i dati nella prima scheda.")
    else:
        try:
            report = requests.get(API_URL + "/report/" + st.session_state.session_id, timeout=30)
            report.raise_for_status()
            report = report.json()
            c1, c2, c3 = st.columns(3)
            c1.metric("Record reali", report["real_rows"])
            c2.metric("Record sintetici", report["synthetic_rows"])
            c3.metric("Copie esatte", report["identical_records"])
            if report["mean_correlation_gap"] is not None:
                st.write(f"Differenza media assoluta di correlazione: **{report['mean_correlation_gap']:.3f}** (più bassa è meglio).")
            st.warning(report["note"])
        except (requests.exceptions.RequestException, ValueError) as exc:
            st.error(f"Report non disponibile: {exc}")
        nums = [c for c in synthetic.select_dtypes(include="number") if c in real]
        if nums:
            key = st.selectbox("Variabile per il confronto distribuzioni", nums)
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.hist(real[key].dropna(), bins=30, alpha=0.5, density=True, label="Reali")
            ax.hist(synthetic[key].dropna(), bins=30, alpha=0.5, density=True, label="Sintetici")
            ax.set(xlabel=key, ylabel="Densità", title="Distribuzioni a confronto")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
        a, b = st.columns(2)
        with a:
            heatmap(real, "Correlazioni reali")
        with b:
            heatmap(synthetic, "Correlazioni sintetiche")
        st.subheader("Confronto tabellare")
        a, b = st.columns(2)
        with a:
            st.caption("Dati originali")
            st.dataframe(real.head(20), use_container_width=True)
        with b:
            st.caption("Dati sintetici")
            st.dataframe(synthetic.head(20), use_container_width=True)

with tab_llm:
    st.write("Chiedi un grafico sui soli **dati sintetici**. Il modello riceve schema e domande, non le righe.")
    if st.session_state.synthetic is None:
        st.info("Genera prima un dataset sintetico.")
    else:
        # Mantieni a video domande e grafici delle precedenti interazioni.
        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.write(msg["content"])
                else:
                    st.write(msg["result"]["title"])
                    st.image(base64.b64decode(msg["result"]["image_base64"]))
                    with st.expander("Codice Pandas/Matplotlib equivalente"):
                        st.code(msg["result"]["example_python"], language="python")
        question = st.chat_input("Es.: qual è la correlazione tra reddito e risparmio?")
        if question:
            previous = [m["content"] for m in st.session_state.chat if m["role"] == "user"][-6:]
            st.session_state.chat.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Analizzo i dati sintetici..."):
                        result = post("/analyze", json={"session_id": st.session_state.session_id,
                            "question": question, "history": previous}).json()
                    st.write(result["title"])
                    st.image(base64.b64decode(result["image_base64"]))
                    with st.expander("Codice Pandas/Matplotlib equivalente"):
                        st.code(result["example_python"], language="python")
                    st.session_state.chat.append({"role": "assistant", "result": result})
                except ValueError as exc:
                    st.error(str(exc))
