import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

# ==============================================================================
# 1. SETUP E CONFIGURAZIONE
# ==============================================================================
st.set_page_config(page_title="Billie AI-lish Live", page_icon="🎧", layout="wide")

# Gestione percorsi per importare moduli locali
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Tentativo importazione motore LSTM
try:
    from lstm_recommender import LSTMRecommender
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

# Percorsi Dati
DATA_DIR = os.path.join(current_dir, '..', 'data')
HISTORY_PATH = os.path.join(DATA_DIR, 'user_history.csv')
DB_PATH = os.path.join(DATA_DIR, 'tracks_processed.csv')

# Palette Colori Sobri (Stile "Architectural")
SOBER_COLORS = {
    "LiveEngine": "#D4AF37",       # Oro Antico (per il live)
    "Background": "#1E1E1E",       # Sfondo Scuro
    "Text": "#E0E0E0"              # Testo Chiaro
}

# ==============================================================================
# 2. CLASSE LOGICA APP (Adattata per Streamlit)
# ==============================================================================
class BillieStreamlitApp:
    """
    Versione Web della BillieAILishApp. 
    Gestisce lo stato della sessione e la logica di raccomandazione.
    """
    def __init__(self):
        self.recommender = None
        if LSTM_AVAILABLE:
            try:
                # Usiamo st.cache_resource per non ricaricare il modello a ogni click
                self.recommender = self._load_engine()
            except Exception as e:
                st.error(f"Errore inizializzazione LSTM: {e}")
        
        # Inizializzazione Session State per Blacklist
        if 'session_blacklist' not in st.session_state:
            st.session_state.session_blacklist = []

    @st.cache_resource
    def _load_engine(_self):
        """Carica il modello pesante in cache."""
        return LSTMRecommender()

    def load_history(self):
        """Carica storico aggiornato."""
        if os.path.exists(HISTORY_PATH):
            return pd.read_csv(HISTORY_PATH)
        return pd.DataFrame()

    def get_mood_details(self, energy, valence):
        """Restituisce etichetta e colore per il mood."""
        if energy > 0.6 and valence > 0.6:
            return "Energico / Felice", "#FF9F1C"
        elif energy > 0.6 and valence < 0.4:
            return "Intenso / Arrabbiato", "#E71D36"
        elif energy < 0.4 and valence > 0.6:
            return "Rilassante / Calmo", "#2EC4B6"
        elif energy < 0.4 and valence < 0.4:
            return "Malinconico / Triste", "#011627"
        else:
            return "Neutro / Bilanciato", "#A0A0A0"

    def run_live_recommendation(self, k=5):
        """Esegue la pipeline di raccomandazione live."""
        history = self.load_history()
        
        if history.empty:
            st.warning("Storico vuoto. Impossibile generare predizioni personalizzate.")
            return

        if not self.recommender:
            st.error("Motore LSTM non disponibile.")
            return

        with st.spinner(f"Analisi ultimi ascolti e generazione di {k} consigli..."):
            # Chiamata al motore di raccomandazione
            recs, predicted_features = self.recommender.recommend(
                user_history_df=history,
                k=k,
                exclude_listened=True,
                session_blacklist=st.session_state.session_blacklist
            )

            # Aggiornamento Blacklist Sessione
            if not recs.empty and 'id' in recs.columns:
                st.session_state.session_blacklist.extend(recs['id'].tolist())

            return recs, predicted_features

# ==============================================================================
# 3. FUNZIONI DI VISUALIZZAZIONE (GRAFICA)
# ==============================================================================
def render_sobrio_card(title, subtitle, badge_text, badge_value, color, footer_text):
    """
    Genera il codice HTML per una card sobria senza indentazioni problematiche.
    """
    # Costruzione stringa unificata per evitare bug di indentazione
    html = (
        f'<div style="background-color: #1E1E1E; color: #E0E0E0; padding: 20px; '
        f'border-radius: 8px; border-left: 5px solid {color}; text-align: left; '
        f'height: 280px; display: flex; flex-direction: column; justify-content: space-between; '
        f'box-shadow: 0 4px 6px rgba(0,0,0,0.3); font-family: sans-serif; margin-bottom: 20px;">'
        f'<div>'
        f'<p style="margin:0; font-size: 0.8em; text-transform: uppercase; letter-spacing: 1px; color: {color}; font-weight: bold;">{title}</p>'
        f'<h2 style="margin: 15px 0 5px 0; font-size: 1.3em; line-height: 1.2; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; color: #FFFFFF;">{subtitle}</h2>'
        f'<p style="margin:0; font-size: 0.95em; opacity: 0.7;">{footer_text}</p>'
        f'</div>'
        f'<div style="border-top: 1px solid #333; padding-top: 15px;">'
        f'<div style="display: flex; justify-content: space-between; align-items: center;">'
        f'<span style="font-size: 0.8em; opacity: 0.6;">{badge_text}</span>'
        f'<span style="font-size: 0.9em; font-weight: bold; color: {color};">{badge_value}</span>'
        f'</div>'
        f'</div>'
        f'</div>'
    )
    st.write(html, unsafe_allow_html=True)

def render_vector_analysis(ideal_vector, real_vector, title, color):
    """Renderizza il grafico vettoriale comparativo (CORRETTO PER EVITARE VALUEERROR)."""
    cols = ['energy', 'valence', 'danceability', 'tempo', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness']
    
    # Adattamento dimensioni se necessario
    length = min(len(ideal_vector), len(real_vector), len(cols))
    ideal_vector = ideal_vector[:length]
    real_vector = real_vector[:length]
    cols = cols[:length]
    
    # Casting esplicito a float per evitare errori di tipo misto in Plotly
    df_comp = pd.DataFrame({
        "Feature": cols,
        "Predizione (AI)": np.array(ideal_vector, dtype=float),
        "Realtà (Brano)": np.array(real_vector, dtype=float)
    })
    
    fig = px.line(df_comp, x="Feature", y=["Predizione (AI)", "Realtà (Brano)"], 
                  markers=True, color_discrete_sequence=[color, "#FFFFFF"])
    fig.update_layout(height=250, margin=dict(l=0,r=0,t=30,b=0), 
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# 4. MAIN APPLICATION LOGIC
# ==============================================================================
def main():
    st.title(" Billie AI-lish: Live Music Engine")
    st.caption("Sistema di raccomandazione musicale in tempo reale basato su architettura LSTM.")
    
    # Inizializza l'app logica
    app_logic = BillieStreamlitApp()

    # --- SIDEBAR DI CONTROLLO ---
    with st.sidebar:
        st.header("Parametri")
        k_recs = st.slider("Nr. Suggerimenti", 3, 12, 6)
        st.divider()
        if st.button("Reset Blacklist Sessione"):
            st.session_state.session_blacklist = []
            st.success("Storico sessione pulito.")

    # --- LIVE RECOMMENDER (LSTM) ---
    
    # Anteprima Storico (Opzionale, collassabile)
    history = app_logic.load_history()
    with st.expander("Visualizza Ultimi Ascolti Utilizzati"):
        if not history.empty:
            st.dataframe(history.tail(5)[['name', 'artist', 'played_at']], use_container_width=True)
        else:
            st.info("Nessuno storico trovato.")

    # Pulsante Azione
    if st.button("Genera Canzoni", type="primary"):
        results = app_logic.run_live_recommendation(k=k_recs)
        
        if results:
            recs, pred_vector = results
            
            st.divider()
            st.subheader(f"Canzoni Generata ({len(recs)} brani)")
            
            # Griglia dinamica per le card (3 per riga)
            grid_cols = st.columns(3)
            
            for idx, row in enumerate(recs.itertuples()):
                col_idx = idx % 3
                
                # Calcolo Dati per UI
                mood_txt, mood_col = app_logic.get_mood_details(row.energy, row.valence)
                match_score = row.match_percentage
                
                with grid_cols[col_idx]:
                    # Render Card Sobria
                    render_sobrio_card(
                        title=f"LSTM #{idx+1} • {mood_txt}",
                        subtitle=row.name,
                        footer_text=row.artist,
                        badge_text="Compatibilità",
                        badge_value=f"{match_score:.1f}%",
                        color=SOBER_COLORS["LiveEngine"] # Oro per il live
                    )
                    
                    # Analisi Vettoriale Reale vs Predetto
                    with st.expander("Dettagli Vettoriali"):
                        # Vettore reale della canzone
                        real_vec = [row.energy, row.valence, row.danceability, row.tempo, row.loudness, 
                                    row.speechiness, row.acousticness, row.instrumentalness, row.liveness]
                        render_vector_analysis(pred_vector, real_vec, "LSTM Live", SOBER_COLORS["LiveEngine"])

    # Info footer
    st.divider()

if __name__ == "__main__":
    main()