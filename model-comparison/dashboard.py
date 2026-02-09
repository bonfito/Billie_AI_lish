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
            subset.style.highlight_min(subset=["MSE (Test)"], color="#1DB954")
                        .highlight_max(subset=["Cosine Similarity"], color="#1DB954")
                        .format({"MSE (Test)": "{:.6f}", "Cosine Similarity": "{:.4f}"}),
            width='stretch',
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
                    fig = px.line(
                        df_hist,
                        x="Epoch",
                        y=["Train Loss", "Test Loss"],markers=True,
                        height=300,
                        color_discrete_sequence=["#A5D6A7", "#1DB954"]
                        )
                    
                    fig.update_layout(
                        legend=dict(orientation="h", y=1.1, title=None),
                        margin=dict(l=0, r=0, t=20, b=0),
                        yaxis_title="Loss (MSE)"
                    )
                    st.plotly_chart(fig, width='stretch')
                    
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
                            .highlight_min(subset=["Test Loss"], color="#1DB954", axis=0), 
                        width='stretch', 
                        height=table_height,
                        hide_index=True
                    )
