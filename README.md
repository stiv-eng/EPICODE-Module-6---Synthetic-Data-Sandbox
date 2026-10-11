# Synthetic Data Sandbox (VAE Edition)

Laboratorio didattico per dati tabellari: preprocessing, training VAE in PyTorch, generazione e validazione CSV, analisi in linguaggio naturale con OpenAI.

## Struttura

- `core.py` — pulizia NaN, MinMaxScaler, OneHotEncoder, encoder/decoder, loss e sampling.
- `api.py` — API FastAPI, sessioni locali in RAM, training, download, report e richieste LLM.
- `ui.py` — interfaccia Streamlit con anteprime, distribuzioni, correlazioni e grafici dell'analista.
- `sample_data.csv` — dataset dimostrativo **inventato** (non contiene dati personali reali).
- `tests/test_project.py` — test minimi del flusso e delle funzioni principali.

## Flusso di utilizzo

1. Carica `sample_data.csv` e premi **Prepara dataset**.
2. Scegli epoche, spazio latente e beta, poi premi **Addestra VAE**.
3. Seleziona il numero di record e premi **Genera dataset sintetico**.
4. Scarica il CSV; nella seconda scheda confronta istogrammi, matrici di correlazione e report.
5. Con una chiave API OpenAI, nella terza scheda chiedi: *Qual è la correlazione tra reddito e risparmio?*

## Endpoint

| Metodo | Route | Funzione |
| --- | --- | --- |
| POST | `/upload` | Riceve CSV via `multipart/form-data`, restituisce `session_id` |
| POST | `/train` | Allena VAE con `session_id`, `epochs`, `latent_dim`, `beta` |
| POST | `/generate` | Restituisce CSV sintetico con `session_id` e `count` |
| GET | `/report/{session_id}` | Confronto correlazioni e copie identiche |
| POST | `/analyze` | Genera grafico Pandas/Matplotlib su richiesta in linguaggio naturale |

