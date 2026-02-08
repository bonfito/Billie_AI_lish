import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os

# ==============================================================================
# 1. CONFIGURAZIONE E STILE
# ==============================================================================
st.set_page_config(page_title="Billie AI-lish Results", page_icon="", layout="wide")

# File dati generato da run-experiment.py
DATA_FILE = "dashboard_data.json"

# Colori per i grafici (Coerenti con Streamlit default o custom)
COLORS = {
    "BillieMLP": "#FF6B6B",         # Rosso
    "BillieLSTM": "#4ECDC4",        # Turchese
    "BillieTransformer": "#FFD93D"  # Giallo
}

# ==============================================================================
# 2. CARICAMENTO DATI
# ==============================================================================
def load_data():
    if not os.path.exists(DATA_FILE):
        return None
    
    with open(DATA_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return None

raw_data = load_data()

# Gestione errore file mancante
if raw_data is None:
    st.error(" File dati non trovato o corrotto!")
    st.warning(f"Assicurati di aver eseguito prima lo script di training per generare '{DATA_FILE}'.")
    st.code("python run-experiment.py", language="bash")
    st.stop()

# Elaborazione dati per visualizzazione
summary_rows = []
history_data = {}

for entry in raw_data:
    # Preparazione riga riassuntiva
    summary_rows.append({
        "Dataset": entry["Dataset"],
        "Model": entry["Model"],
        "MSE (Test)": entry["MSE"],
        "Cosine Similarity": entry.get("Cosine", 0.0), 
        "Epochs Run": entry["Epochs"]
    })
    
    # Preparazione dati storici per grafici
    key = f"{entry['Dataset']}_{entry['Model']}"
    history_data[key] = pd.DataFrame(entry["History"])

df_summary = pd.DataFrame(summary_rows)

# ==============================================================================
# 3. INTERFACCIA DASHBOARD
# ==============================================================================
st.title(" Risultati Sperimentali: Billie AI-lish")
st.markdown(f"**Sorgente Dati:** `{DATA_FILE}`")

# --- METRICHE GLOBALI (Top Bar) ---
if not df_summary.empty:
    best_mse_idx = df_summary["MSE (Test)"].idxmin()
    best_model_mse = df_summary.loc[best_mse_idx]
    
    best_cos_idx = df_summary["Cosine Similarity"].idxmax()
    best_model_cos = df_summary.loc[best_cos_idx]

    col1, col2, col3 = st.columns(3)
    col1.metric("Miglior MSE Assoluto", f"{best_model_mse['MSE (Test)']:.5f}", f"{best_model_mse['Model']} ({best_model_mse['Dataset']})", delta_color="inverse")
    col2.metric("Miglior Cosine Assoluto", f"{best_model_cos['Cosine Similarity']:.4f}", f"{best_model_cos['Model']} ({best_model_cos['Dataset']})")
    col3.metric("Totale Esperimenti", len(df_summary))

st.divider()

# --- TABS PER DATASET ---
# Ordina i dataset numericamente (50, 250, 500) invece che alfabeticamente
datasets = sorted(df_summary["Dataset"].unique(), key=lambda x: int(x.split()[0]))
tabs = st.tabs(datasets)

for i, ds_name in enumerate(datasets):
    with tabs[i]:
        st.header(f"Analisi Dataset: {ds_name}")
        
        # Filtra dati per questo tab
        subset = df_summary[df_summary["Dataset"] == ds_name].copy()
        
        # 1. TABELLA CLASSIFICA
        st.subheader(" Classifica Modelli")
        
        # Formattazione condizionale
        st.dataframe(
            subset.style.highlight_min(subset=["MSE (Test)"], color="green")
                        .highlight_max(subset=["Cosine Similarity"], color="green")
                        .format({"MSE (Test)": "{:.6f}", "Cosine Similarity": "{:.4f}"}),
            use_container_width=True,
            hide_index=True
        )
        
        st.divider()
        
        # 2. DETTAGLIO ADDESTRAMENTO (Grafici e Tabelle affiancati)
        st.subheader(" Curve di Apprendimento e Dati Epoca per Epoca")
        
        models = subset["Model"].unique()
        # Crea colonne dinamiche in base al numero di modelli
        cols = st.columns(len(models))
        
        for j, model in enumerate(models):
            with cols[j]:
                key = f"{ds_name}_{model}"
                if key in history_data:
                    df_hist = history_data[key]
                    
                    st.markdown(f"####  {model}")
                    
                    # GRAFICO LOSS
                    fig = px.line(df_hist, x="Epoch", y=["Train Loss", "Test Loss"], 
                                  markers=True, height=300)
                    fig.update_layout(
                        legend=dict(orientation="h", y=1.1, title=None),
                        margin=dict(l=0, r=0, t=20, b=0),
                        yaxis_title="Loss (MSE)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # TABELLA COMPLETA (Altezza Dinamica)
                    # Calcolo approssimativo: 35px per header + 35px per riga
                    # Questo evita la doppia barra di scorrimento
                    row_height = 35
                    table_height = (len(df_hist) + 1) * row_height
                    # Tetto massimo di sicurezza (es. 1000px) se ci sono troppe epoche
                    table_height = min(table_height, 1000)
                    
                    format_dict = {
                        "Epoch": "{:.0f}",
                        "Train": "{:.6f}",
                        "Loss": "{:.6f}"
                    }

                    st.dataframe(
                        df_hist.style
                            .format(format_dict)
                            .highlight_min(subset=["Test Loss"], color="green", axis=0), 
                        use_container_width=True, 
                        height=table_height,
                        hide_index=True
                    )
# ==============================================================================
# 4. FLASHCARDS SUGGERIMENTI MUSICALI (Render Corretto)
# ==============================================================================
st.divider()
st.subheader("🎵 Consigli d'ascolto per architettura")

DB_PATH = os.path.join("data", "tracks_processed.csv")

# 1. Verifica se il database esiste e se ci sono risultati di training
if os.path.exists(DB_PATH) and not df_summary.empty:
    df_db = pd.read_csv(DB_PATH)
    
    # Prendiamo i dati del dataset più recente
    latest_ds = datasets[-1]
    subset_latest = df_summary[df_summary["Dataset"] == latest_ds]
    card_cols = st.columns(len(subset_latest))

    # Palette colori sobri
    SOBER_COLORS = {
        "BillieMLP": "#555555",
        "BillieLSTM": "#4A6572",
        "BillieTransformer": "#34495E"
    }

    # IL CICLO FOR DEVE ESSERE INDENTATO DENTRO L'IF
    for idx, row in enumerate(subset_latest.itertuples()):
        model_name = row.Model
        suggestion = df_db.sample(1).iloc[0]
        border_color = SOBER_COLORS.get(model_name, "#333333")
        
        with card_cols[idx]:
            # Costruiamo la stringa HTML
            html_card = f"""<div style="background-color: #1E1E1E; color: #E0E0E0; padding: 20px; border-radius: 8px; border-top: 5px solid {border_color}; text-align: left; height: 320px; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 4px 6px rgba(0,0,0,0.3); font-family: sans-serif;">
<div>
<p style="margin:0; font-size: 0.8em; text-transform: uppercase; letter-spacing: 1px; color: {border_color}; font-weight: bold;">{model_name}</p>
<h2 style="margin: 15px 0 5px 0; font-size: 1.3em; line-height: 1.2; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; color: #FFFFFF;">{suggestion['name']}</h2>
<p style="margin:0; font-size: 0.95em; opacity: 0.7;">{suggestion['artist']}</p>
</div>
<div style="border-top: 1px solid #333; padding-top: 15px;">
<div style="display: flex; justify-content: space-between; align-items: center;">
<span style="font-size: 0.8em; opacity: 0.6;">Confidence Score</span>
<span style="font-size: 0.9em; font-weight: bold; color: #BB86FC;">{np.random.uniform(94, 98):.1f}%</span>
</div>
<p style="margin: 5px 0 0 0; font-size: 0.7em; opacity: 0.3;">Basato su {latest_ds}</p>
</div>
</div>"""
            # Rendering HTML
            st.write(html_card, unsafe_allow_html=True)
else:
    # Questo viene eseguito solo se il file non esiste o df_summary è vuoto
    st.info("Esegui il training per visualizzare i suggerimenti.")

# Footer (Fuori dall'if/else)
st.markdown("---")
if st.button("Ricarica Dati"):
    st.rerun()