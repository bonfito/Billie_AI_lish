"""
streamlit_app.py

Interfaccia Streamlit per LSTM Music Recommender.
Design moderno, interattivo, visualizzazioni grafiche.

Autore: Gaetano
Data: Febbraio 2026

Uso: streamlit run streamlit_app.py
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
project_root = os.path.dirname(current_dir)  # Risale di un livello per trovare .env
dotenv_path = os.path.join(project_root, '.env')

# Carica variabili d'ambiente
load_dotenv(dotenv_path)

# Aggiungi directory corrente al path per importare moduli locali
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from lstm_recommender import LSTMRecommender


# ═══════════════════════════════════════════════════════════════════
# CONFIGURAZIONE PAGINA
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Billie AI-lish 🎵",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Custom
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        width: 100%;
    }
    
    .stButton>button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# INIZIALIZZAZIONE STATE
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource
def load_recommender():
    """Carica recommender (cache per non ricaricare ad ogni interazione)."""
    try:
        return LSTMRecommender()
    except Exception as e:
        st.error(f"❌ Errore caricamento recommender: {e}")
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

# Inizializza session state
if 'session_blacklist' not in st.session_state:
    st.session_state.session_blacklist = []

if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None

if 'predicted_features' not in st.session_state:
    st.session_state.predicted_features = None


# ═══════════════════════════════════════════════════════════════════
# FUNZIONI UTILITY
# ═══════════════════════════════════════════════════════════════════

def get_mood_emoji(energy, valence):
    """Determina emoji mood da energy e valence."""
    if energy > 0.6 and valence > 0.6:
        return "🔥", "Energico"
    elif energy > 0.6 and valence < 0.4:
        return "⚡", "Intenso"
    elif energy < 0.4 and valence > 0.6:
        return "🌊", "Rilassante"
    elif energy < 0.4 and valence < 0.4:
        return "🌧️", "Malinconico"
    else:
        return "😌", "Neutro"

def create_radar_chart(features, feature_names):
    """Crea radar chart per audio features."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=features,
        theta=feature_names,
        fill='toself',
        name='Predizione LSTM',
        line=dict(color='#667eea', width=2),
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showticklabels=True,
                ticks='',
                gridcolor='lightgray'
            )
        ),
        showlegend=True,
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12)
    )
    
    return fig

def create_history_chart(user_history, feature='energy'):
    """Crea grafico trend feature nel tempo."""
    if user_history.empty or feature not in user_history.columns:
        return None
    
    recent = user_history.tail(50).copy()
    recent['index'] = range(len(recent))
    
    fig = px.line(
        recent,
        x='index',
        y=feature,
        title=f'Trend {feature.capitalize()} (Ultime 50 Canzoni)',
        labels={'index': 'Canzone', feature: feature.capitalize()},
        markers=True
    )
    
    fig.update_traces(
        line=dict(color='#667eea', width=3),
        marker=dict(size=6)
    )
    
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='lightgray')
    )
    
    return fig

def create_comparison_chart(history_mean, predicted, feature_names):
    """Crea bar chart confronto storico vs predizione."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Media Storico',
        x=feature_names,
        y=history_mean,
        marker_color='#95a5a6'
    ))
    
    fig.add_trace(go.Bar(
        name='Predizione LSTM',
        x=feature_names,
        y=predicted,
        marker_color='#667eea'
    ))
    
    fig.update_layout(
        barmode='group',
        height=400,
        title='Confronto: Storico vs Predizione LSTM',
        yaxis=dict(range=[0, 1]),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis_title='Valore',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════

st.markdown('<h1 class="main-header">🎵 Billie AI-lish</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align: center; font-size: 1.2rem; color: #666; margin-top: -1rem;">'
    'Music Recommender basato su <strong>LSTM Deep Learning</strong>'
    '</p>',
    unsafe_allow_html=True
)

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR - CONTROLLI
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Configurazione")
    
    # Info sistema
    with st.expander("📊 Info Sistema", expanded=False):
        recommender = load_recommender()
        user_history = load_user_history()
        
        st.metric("Modello", "BillieLSTM")
        st.metric("Database Canzoni", f"{len(recommender.df_tracks):,}")
        st.metric("Storico Utente", f"{len(user_history)}")
        st.metric("Device", str(recommender.device))
    
    st.markdown("---")
    
    # Numero raccomandazioni
    st.subheader("🎯 Raccomandazioni")
    k_recommendations = st.slider(
        "Numero canzoni",
        min_value=5,
        max_value=50,
        value=20,
        step=5
    )
    
    # Filtri
    st.subheader("🔍 Filtri")
    exclude_listened = st.checkbox(
        "Escludi già ascoltate",
        value=True,
        help="Rimuove canzoni presenti nello storico"
    )
    
    # Reset blacklist
    if st.button("🔄 Reset Blacklist Sessione"):
        st.session_state.session_blacklist = []
        st.success("✅ Blacklist resettata!")
    
    st.caption(f"🚫 Blacklist: {len(st.session_state.session_blacklist)} canzoni")
    
    st.markdown("---")
    
    # Bottone genera
    generate_button = st.button(
        "🎵 Genera Raccomandazioni",
        type="primary",
        use_container_width=True
    )


# ═══════════════════════════════════════════════════════════════════
# MAIN - TABS
# ═══════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "🎵 Raccomandazioni",
    "📊 Analisi LSTM",
    "📜 Storico",
    "ℹ️ Info"
])


# ───────────────────────────────────────────────────────────────────
# TAB 1: RACCOMANDAZIONI
# ───────────────────────────────────────────────────────────────────

with tab1:
    if generate_button or st.session_state.recommendations is not None:
        
        # Genera raccomandazioni
        if generate_button:
            with st.spinner("🎵 Generazione raccomandazioni in corso..."):
                recommender = load_recommender()
                user_history = load_user_history()
                
                recommendations, predicted = recommender.recommend(
                    user_history_df=user_history,
                    k=k_recommendations,
                    exclude_listened=exclude_listened,
                    session_blacklist=st.session_state.session_blacklist
                )
                
                # Salva in session state
                st.session_state.recommendations = recommendations
                st.session_state.predicted_features = predicted
                
                # Aggiungi a blacklist
                if not recommendations.empty and 'id' in recommendations.columns:
                    new_ids = recommendations['id'].tolist()
                    st.session_state.session_blacklist.extend(new_ids)
        
        # Mostra raccomandazioni
        recommendations = st.session_state.recommendations
        
        if recommendations is not None and not recommendations.empty:
            st.success(f"✅ {len(recommendations)} raccomandazioni generate!")
            
            # Metrics in colonne
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_match = recommendations['match_percentage'].mean()
                st.metric("Match Medio", f"{avg_match:.1f}%")
            
            with col2:
                top_match = recommendations['match_percentage'].iloc[0]
                st.metric("Top Match", f"{top_match:.1f}%")
            
            with col3:
                unique_artists = recommendations['artist'].nunique() if 'artist' in recommendations.columns else 0
                st.metric("Artisti Diversi", unique_artists)
            
            st.markdown("---")
            
            # Lista raccomandazioni
            st.subheader("🎵 Le Tue Raccomandazioni")
            
            for idx, row in recommendations.iterrows():
                rank = row.get('rank', idx + 1)
                name = row.get('name', 'Unknown')
                artist = row.get('artist', 'Unknown')
                score = row.get('match_percentage', 0)
                track_id = row.get('id', None)
                
                # Mood
                energy = row.get('energy', 0.5)
                valence = row.get('valence', 0.5)
                emoji, mood_text = get_mood_emoji(energy, valence)
                
                # Card canzone (Unico contenitore, niente colonne separate per copertina)
                with st.container():
                    
                    # 1. Intestazione Metadati (Rank, Match, Mood)
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 15px; margin-bottom: 5px;">
                            <div>
                                <span style="font-size: 1.1rem; font-weight: bold;">#{rank}</span>
                                <span style="font-size: 0.9rem; background-color: #f0f2f6; padding: 2px 8px; border-radius: 10px; margin-left: 8px;">{emoji} {mood_text}</span>
                            </div>
                            <div style="font-size: 0.9rem; color: #667eea; font-weight: bold;">
                                Match: {score:.1f}%
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # 2. Player Spotify Standard (include copertina)
                    if pd.notna(track_id) and track_id:
                        spotify_embed_url = f"https://open.spotify.com/embed/track/{track_id}?utm_source=generator&theme=0"
                        
                        # height=152 mostra il player "grande" con copertina visibile
                        st.markdown(
                            f"""
                            <iframe 
                                style="border-radius:12px" 
                                src="{spotify_embed_url}" 
                                width="100%" 
                                height="152" 
                                frameBorder="0" 
                                allowfullscreen="" 
                                allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" 
                                loading="lazy">
                            </iframe>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.warning(f"⚠️ {name} - {artist} (Anteprima non disponibile)")
                    
                    # 3. Audio features (collapsible)
                    with st.expander("📊 Dettagli Audio"):
                        audio_cols = ['energy', 'valence', 'danceability', 'tempo']
                        available = [c for c in audio_cols if c in row.index]
                        
                        if available:
                            feat_cols = st.columns(len(available))
                            for i, feat in enumerate(available):
                                with feat_cols[i]:
                                    st.metric(
                                        feat.capitalize(),
                                        f"{row[feat]:.2f}"
                                    )
        else:
            st.warning("⚠️ Nessuna raccomandazione disponibile")
    
    else:
        # Messaggio iniziale
        st.info(
            "👈 Usa la sidebar per configurare e generare raccomandazioni!\n\n"
            "Il sistema userà **LSTM** per predire le audio features della tua prossima "
            "canzone preferita basandosi sullo storico."
        )


# ───────────────────────────────────────────────────────────────────
# TAB 2: ANALISI LSTM
# ───────────────────────────────────────────────────────────────────

with tab2:
    st.header("📊 Analisi Predizione LSTM")
    
    user_history = load_user_history()
    
    if user_history.empty:
        st.warning("⚠️ Nessuno storico disponibile per l'analisi")
    else:
        recommender = load_recommender()
        
        # Analizza predizione
        analysis = recommender.analyze_prediction(user_history)
        predicted = analysis['predicted']
        
        # Radar chart predizione
        st.subheader("🎯 Profilo Predizione LSTM")
        
        radar_fig = create_radar_chart(
            predicted,
            recommender.audio_features
        )
        st.plotly_chart(radar_fig, use_container_width=True)
        
        # Confronto storico vs predizione
        if analysis['history_mean'] is not None:
            st.markdown("---")
            st.subheader("📊 Confronto: Storico vs Predizione")
            
            comparison_fig = create_comparison_chart(
                analysis['history_mean'],
                predicted[:len(analysis['feature_names'])],
                analysis['feature_names']
            )
            st.plotly_chart(comparison_fig, use_container_width=True)
            
            # Tabella dettaglio
            st.markdown("---")
            st.subheader("📋 Dettaglio Numerico")
            
            df_comparison = pd.DataFrame({
                'Feature': analysis['feature_names'],
                'Media Storico': [f"{x:.3f}" for x in analysis['history_mean']],
                'Predizione LSTM': [f"{x:.3f}" for x in predicted[:len(analysis['feature_names'])]],
                'Differenza': [f"{x:+.3f}" for x in analysis['difference']],
                'Trend': ['↑ Aumento' if x > 0.05 else '↓ Diminuzione' if x < -0.05 else '→ Stabile' 
                         for x in analysis['difference']]
            })
            
            st.dataframe(
                df_comparison,
                use_container_width=True,
                hide_index=True
            )
            
            # Interpretazione
            st.markdown("---")
            st.subheader("💡 Interpretazione")
            
            # Energy
            if 'energy' in analysis['feature_names']:
                idx = analysis['feature_names'].index('energy')
                energy_diff = analysis['difference'][idx]
                
                if energy_diff > 0.1:
                    st.info(
                        "🔥 **LSTM prevede AUMENTO di energia**\n\n"
                        "Le raccomandazioni saranno più energiche del tuo storico."
                    )
                elif energy_diff < -0.1:
                    st.info(
                        "🌊 **LSTM prevede DIMINUZIONE di energia**\n\n"
                        "Le raccomandazioni saranno più rilassate del tuo storico."
                    )
                else:
                    st.info(
                        "😌 **LSTM prevede energia STABILE**\n\n"
                        "Le raccomandazioni manterranno l'energia del tuo storico."
                    )
            
            # Valence
            if 'valence' in analysis['feature_names']:
                idx = analysis['feature_names'].index('valence')
                valence_diff = analysis['difference'][idx]
                
                if valence_diff > 0.1:
                    st.success(
                        "😊 **LSTM prevede AUMENTO di positività**\n\n"
                        "Le raccomandazioni saranno più allegre."
                    )
                elif valence_diff < -0.1:
                    st.warning(
                        "😔 **LSTM prevede DIMINUZIONE di positività**\n\n"
                        "Le raccomandazioni saranno più malinconiche."
                    )


# ───────────────────────────────────────────────────────────────────
# TAB 3: STORICO
# ───────────────────────────────────────────────────────────────────

with tab3:
    st.header("📜 Storico Ascolti")
    
    user_history = load_user_history()
    
    if user_history.empty:
        st.warning("⚠️ Nessuno storico disponibile")
    else:
        # Statistiche
        st.subheader("📊 Statistiche Generali")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Totale Canzoni", len(user_history))
        
        with col2:
            unique_artists = user_history['artist'].nunique() if 'artist' in user_history.columns else 0
            st.metric("Artisti Unici", unique_artists)
        
        with col3:
            if 'energy' in user_history.columns:
                avg_energy = user_history['energy'].mean()
                st.metric("Energia Media", f"{avg_energy:.2f}")
        
        with col4:
            if 'valence' in user_history.columns:
                avg_valence = user_history['valence'].mean()
                st.metric("Positività Media", f"{avg_valence:.2f}")
        
        st.markdown("---")
        
        # Trend features
        st.subheader("📈 Trend Audio Features")
        
        feature_to_plot = st.selectbox(
            "Seleziona feature da visualizzare",
            options=['energy', 'valence', 'danceability', 'tempo', 'loudness'],
            index=0
        )
        
        trend_fig = create_history_chart(user_history, feature_to_plot)
        if trend_fig:
            st.plotly_chart(trend_fig, use_container_width=True)
        
        st.markdown("---")
        
        # Tabella ultime 20
        st.subheader("🎵 Ultime 20 Canzoni Ascoltate")
        
        display_cols = ['name', 'artist', 'energy', 'valence']
        available_cols = [c for c in display_cols if c in user_history.columns]
        
        if available_cols:
            recent = user_history[available_cols].tail(20).copy()
            recent.index = range(len(recent), 0, -1)
            
            st.dataframe(
                recent,
                use_container_width=True,
                height=400
            )


# ───────────────────────────────────────────────────────────────────
# TAB 4: INFO
# ───────────────────────────────────────────────────────────────────

with tab4:
    st.header("ℹ️ Informazioni Sistema")
    
    # Come funziona
    with st.expander("🎯 Come Funziona", expanded=True):
        st.markdown("""
        ### Pipeline di Raccomandazione
        
        1. **📂 Caricamento Storico** Sistema carica le tue ultime 20 canzoni ascoltate
        
        2. **🧠 Predizione LSTM** Il modello **BillieLSTM** analizza le audio features delle canzoni  
           e predice le caratteristiche della tua prossima canzone preferita
        
        3. **🔍 Ricerca Similarità** Calcola la **similarità coseno** tra la predizione e 2.8M canzoni nel database
        
        4. **🎵 Ranking** Ordina le canzoni per similarità e restituisce le top K
        
        5. **🚫 Filtri** Rimuove canzoni già ascoltate e blacklist sessione
        """)
    
    # Audio Features
    with st.expander("📊 Audio Features (9 Dimensioni)"):
        st.markdown("""
        | Feature | Descrizione | Range |
        |---------|-------------|-------|
        | **energy** | Intensità/Attività | 0-1 |
        | **valence** | Positività emotiva | 0-1 |
        | **danceability** | Ballabilità | 0-1 |
        | **tempo** | BPM normalizzato | 0-1 |
        | **loudness** | Volume normalizzato | 0-1 |
        | **speechiness** | Voce parlata | 0-1 |
        | **acousticness** | Acusticità | 0-1 |
        | **instrumentalness** | Strumentale vs vocale | 0-1 |
        | **liveness** | Registrato dal vivo | 0-1 |
        """)
    
    # Performance
    with st.expander("🏆 Performance Modello"):
        st.markdown("""
        ### Risultati Esperimento
        
        Il modello **BillieLSTM** è stato addestrato e testato su 3 dataset:
        
        | Dataset | MSE | Cosine Similarity | Vincitore |
        |---------|-----|-------------------|-----------|
        | 50 canzoni | 0.015989 | 0.9779 | 🏆 LSTM |
        | 250 canzoni | 0.029747 | 0.9461 | 🏆 LSTM |
        | 500 canzoni | 0.028417 | 0.9518 | 🏆 LSTM |
        
        **LSTM domina su tutte le configurazioni!**
        
        Confronto con architetture alternative:
        - **LSTM**: MSE 0.028 ✅ **Migliore**
        - Oracle (MLP): MSE 0.030
        - Transformer: MSE 0.029
        """)
    
    # Tecnico
    with st.expander("⚙️ Dettagli Tecnici"):
        recommender = load_recommender()
        
        st.markdown(f"""
        ### Configurazione
        
        - **Modello**: BillieLSTM (204,425 parametri)
        - **Input**: Sequenze di 20 canzoni × 9 features
        - **Output**: Vettore 9D (predizione prossima canzone)
        - **Device**: {recommender.device}
        - **Database**: {len(recommender.df_tracks):,} canzoni
        - **Metrica**: Cosine Similarity
        
        ### Architettura LSTM
        
        ```python
        BillieLSTM(
            input_size=9,
            hidden_size=128,
            num_layers=2,
            dropout=0.2
        )
        ```
        
        ### File Modello
        
        `{os.path.basename(recommender.model_path)}`
        """)
    
    # Credits
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <h3>🎵 Billie AI-lish</h3>
        <p>Developed by <strong>Gaetano</strong></p>
        <p>Progetto Tesi - Febbraio 2026</p>
        <p style="font-size: 0.9rem;">
            Powered by <strong>LSTM Deep Learning</strong> 🧠<br>
            Built with <strong>Streamlit</strong> ❤️
        </p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🎵 Audio-only recommendations")

with col2:
    st.caption("🧠 LSTM-powered predictions")

with col3:
    st.caption("📊 2.8M songs database")