"""
streamlit_app.py

Interfaccia Streamlit per LSTM Music Recommender.
Design moderno, interattivo, visualizzazioni grafiche.
Stile: Minimale (No Icone).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
from dotenv import load_dotenv

# ═══════════════════════════════════════════════════════════════════
# CONFIGURAZIONE PATH E ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════

# Determina directory corrente e root progetto
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
dotenv_path = os.path.join(project_root, '.env')

# Carica variabili d'ambiente
load_dotenv(dotenv_path)

# Aggiungi directory corrente al path per importare moduli locali
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Gestione import sicuro
try:
    from lstm_recommender import LSTMRecommender
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════
# CONFIGURAZIONE PAGINA
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Billie AI-lish Live",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Custom per layout pulito
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #D4AF37 0%, #E0E0E0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    div.stButton > button {
        background-color: #D4AF37;
        color: white;
        border-radius: 8px;
        border: none;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# FUNZIONI DI SUPPORTO GRAFICO
# ═══════════════════════════════════════════════════════════════════

def render_vector_analysis(ideal_vector, real_vector, color="#D4AF37"):
    """Renderizza il grafico vettoriale comparativo per ogni card."""
    cols = ['energy', 'valence', 'danceability', 'tempo', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness']
    
    # Casting a float per evitare errori Plotly
    ideal_vector = np.array(ideal_vector, dtype=float)
    real_vector = np.array(real_vector, dtype=float)
    
    # Adattamento dimensioni
    length = min(len(ideal_vector), len(real_vector), len(cols))
    
    df_comp = pd.DataFrame({
        "Feature": cols[:length],
        "Target (AI)": ideal_vector[:length],
        "Brano (Reale)": real_vector[:length]
    })
    
    fig = px.line(df_comp, x="Feature", y=["Target (AI)", "Brano (Reale)"], 
                  markers=True, color_discrete_sequence=[color, "#FFFFFF"])
    fig.update_layout(height=250, margin=dict(l=0,r=0,t=20,b=0), 
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend_title_text="", showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

def render_sobrio_card(rank, title, artist, match_score, mood_text, color="#D4AF37"):
    """HTML Card Sobria (Senza Icone, Stile Architetturale)."""
    html = (
        f'<div style="background-color: #1E1E1E; color: #E0E0E0; padding: 20px; '
        f'border-radius: 8px; border-left: 5px solid {color}; text-align: left; '
        f'height: 280px; display: flex; flex-direction: column; justify-content: space-between; '
        f'box-shadow: 0 4px 6px rgba(0,0,0,0.3); font-family: sans-serif; margin-bottom: 10px;">'
        f'<div>'
        f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">'
        f'<span style="font-size: 0.8em; text-transform: uppercase; letter-spacing: 1px; color: {color}; font-weight: bold;">#{rank} LSTM REC</span>'
        f'<span style="font-size: 0.7em; color: #888; border: 1px solid #555; padding: 2px 6px; border-radius: 4px;">{mood_text}</span>'
        f'</div>'
        f'<h2 style="margin: 0 0 5px 0; font-size: 1.3em; line-height: 1.2; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; color: #FFFFFF;">{title}</h2>'
        f'<p style="margin:0; font-size: 0.95em; opacity: 0.7;">{artist}</p>'
        f'</div>'
        f'<div style="border-top: 1px solid #333; padding-top: 15px;">'
        f'<div style="display: flex; justify-content: space-between; align-items: center;">'
        f'<span style="font-size: 0.8em; opacity: 0.6;">Compatibilità</span>'
        f'<span style="font-size: 0.9em; font-weight: bold; color: {color};">{match_score:.1f}%</span>'
        f'</div>'
        f'</div>'
        f'</div>'
    )
    st.write(html, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# INIZIALIZZAZIONE STATE
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource
def load_recommender():
    """Carica recommender (cache per non ricaricare ad ogni interazione)."""
    try:
        return LSTMRecommender()
    except Exception as e:
        st.error(f"Errore caricamento recommender: {e}")
        st.stop()

@st.cache_data
def load_user_history():
    """Carica storico utente."""
    data_dir = os.path.join(current_dir, '..', 'data')
    history_path = os.path.join(data_dir, 'user_history.csv')
    
    if os.path.exists(history_path):
        return pd.read_csv(history_path)
    else:
        return pd.DataFrame()

if 'session_blacklist' not in st.session_state:
    st.session_state.session_blacklist = []

if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None

if 'predicted_features' not in st.session_state:
    st.session_state.predicted_features = None


# ═══════════════════════════════════════════════════════════════════
# FUNZIONI UTILITY
# ═══════════════════════════════════════════════════════════════════

def get_mood_text(energy, valence):
    """Determina testo mood (Senza Emoji)."""
    if energy > 0.6 and valence > 0.6:
        return "Energico"
    elif energy > 0.6 and valence < 0.4:
        return "Intenso"
    elif energy < 0.4 and valence > 0.6:
        return "Rilassante"
    elif energy < 0.4 and valence < 0.4:
        return "Malinconico"
    else:
        return "Neutro"

def create_radar_chart(features, feature_names):
    """Crea radar chart per audio features."""
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=features,
        theta=feature_names,
        fill='toself',
        name='Predizione LSTM',
        line=dict(color='#D4AF37', width=2),
        fillcolor='rgba(212, 175, 55, 0.3)'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=True, ticks='', gridcolor='#444'),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=False,
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0')
    )
    return fig

# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════

st.markdown('<h1 class="main-header">Billie AI-lish Live</h1>', unsafe_allow_html=True)
st.caption("Sistema di raccomandazione sequenziale basato su LSTM Deep Learning.")
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR - CONTROLLI
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("Configurazione")
    
    if LSTM_AVAILABLE:
        recommender = load_recommender()
        user_history = load_user_history()
        st.success(f"Motore LSTM Attivo")
        st.metric("Database", f"{len(recommender.df_tracks):,}")
    else:
        st.error("Motore LSTM non trovato.")
    
    st.markdown("---")
    
    # Parametri
    k_recommendations = st.slider("Numero canzoni", 3, 21, 6, step=3)
    exclude_listened = st.checkbox("Escludi già ascoltate", value=True)
    
    if st.button("Reset Blacklist Sessione"):
        st.session_state.session_blacklist = []
        st.success("Blacklist pulita.")
    
    st.caption(f"Blacklist: {len(st.session_state.session_blacklist)}")
    st.markdown("---")
    
    # Bottone genera
    generate_button = st.button("Genera Playlist", type="primary", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# MAIN - TABS
# ═══════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs(["Playlist Live", "Analisi Predizione", "Storico Dati"])

# ───────────────────────────────────────────────────────────────────
# TAB 1: RACCOMANDAZIONI (STILE FLASHCARD SOBRIO)
# ───────────────────────────────────────────────────────────────────

with tab1:
    if generate_button or st.session_state.recommendations is not None:
        
        # Logica Generazione
        if generate_button and LSTM_AVAILABLE:
            with st.spinner("Analisi sequenza temporale e generazione predizione..."):
                recommender = load_recommender()
                user_history = load_user_history()
                
                recommendations, predicted = recommender.recommend(
                    user_history_df=user_history,
                    k=k_recommendations,
                    exclude_listened=exclude_listened,
                    session_blacklist=st.session_state.session_blacklist
                )
                
                st.session_state.recommendations = recommendations
                st.session_state.predicted_features = predicted
                
                if not recommendations.empty and 'id' in recommendations.columns:
                    st.session_state.session_blacklist.extend(recommendations['id'].tolist())
        
        # Visualizzazione
        recommendations = st.session_state.recommendations
        predicted_vector = st.session_state.predicted_features
        
        if recommendations is not None and not recommendations.empty:
            st.success(f"Playlist generata con successo ({len(recommendations)} brani)")
            
            # --- LAYOUT A GRIGLIA PER LE CARD ---
            cols = st.columns(3) # 3 Card per riga
            
            for idx, row in enumerate(recommendations.itertuples()):
                col_idx = idx % 3
                
                # Dati Card
                rank = idx + 1
                mood_text = get_mood_text(row.energy, row.valence)
                
                with cols[col_idx]:
                    # 1. Flashcard Sobria HTML
                    render_sobrio_card(
                        rank=rank,
                        title=row.name,
                        artist=row.artist,
                        match_score=row.match_percentage,
                        mood_text=mood_text,
                        color="#D4AF37" # Oro per coerenza
                    )
                    
                    # 2. Analisi Vettoriale (Expander sotto ogni card)
                    with st.expander("Analisi Vettoriale"):
                        real_vec = [row.energy, row.valence, row.danceability, row.tempo, row.loudness, 
                                    row.speechiness, row.acousticness, row.instrumentalness, row.liveness]
                        # Usa il vettore predetto globale (uguale per tutti in questa batch) vs il reale
                        render_vector_analysis(predicted_vector, real_vec)
                        
        else:
            if LSTM_AVAILABLE:
                st.warning("Nessuna raccomandazione trovata con i filtri attuali.")
            else:
                st.error("Motore LSTM non disponibile.")
    
    else:
        st.info("Usa il pannello laterale per generare la tua prima playlist AI.")


# ───────────────────────────────────────────────────────────────────
# TAB 2: ANALISI LSTM (RADAR E TREND)
# ───────────────────────────────────────────────────────────────────

with tab2:
    st.header("Analisi del Vettore Predetto")
    
    if st.session_state.predicted_features is not None:
        recommender = load_recommender()
        predicted = st.session_state.predicted_features
        
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.subheader("Radar Chart")
            radar_fig = create_radar_chart(predicted, recommender.audio_features)
            st.plotly_chart(radar_fig, use_container_width=True)
            
        with c2:
            st.subheader("Valori Numerici Target")
            df_pred = pd.DataFrame([predicted], columns=recommender.audio_features)
            st.dataframe(df_pred.style.format("{:.3f}"), use_container_width=True)
            
            st.markdown("### Interpretazione")
            st.info("Questo vettore rappresenta la 'canzone ideale' calcolata matematicamente dalla rete neurale basandosi sui tuoi ascolti precedenti.")
            
    else:
        st.info("Genera una playlist per visualizzare l'analisi della predizione.")


# ───────────────────────────────────────────────────────────────────
# TAB 3: STORICO
# ───────────────────────────────────────────────────────────────────

with tab3:
    st.header("Storico Dati Utente")
    user_history = load_user_history()
    
    if not user_history.empty:
        st.dataframe(user_history.tail(20)[['name', 'artist', 'played_at', 'energy', 'valence']], use_container_width=True)
    else:
        st.warning("Storico vuoto.")

if __name__ == "__main__":
    pass