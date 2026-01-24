import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
from recommender import SongRecommender
from oracle import MusicOracle 
from utils import calculate_avalanche_context

# --- CONFIGURAZIONE PERCORSI ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH_DATA = os.path.normpath(os.path.join(CURRENT_DIR, '..', 'data', 'tracks_db.csv'))
DB_PATH_SRC = os.path.normpath(os.path.join(CURRENT_DIR, 'tracks_db.csv'))
HISTORY_PATH = os.path.normpath(os.path.join(CURRENT_DIR, '..', 'data', 'user_history.csv'))

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Billie AI-lish", layout="wide")
st.title("🎵 Billie AI-lish: Music Discovery Agent")

# --- INIZIALIZZAZIONE ---
if 'oracle' not in st.session_state:
    st.session_state.oracle = MusicOracle() 
    
    try:
        if os.path.exists(DB_PATH_DATA):
            target_db = DB_PATH_DATA
        elif os.path.exists(DB_PATH_SRC):
            target_db = DB_PATH_SRC
        else:
            raise FileNotFoundError(f"Non trovo tracks_db.csv")

        st.session_state.recommender = SongRecommender(target_db)
        st.success(f"✅ Database caricato correttamente")
    except Exception as e:
        st.error(f"⚠️ Errore Database: {e}")
        st.stop()

    # 2. WARM START: Carica Storia Utente (Inclusi Artist & Genre)
    features_needed = st.session_state.recommender.features
    
    try:
        if os.path.exists(HISTORY_PATH):
            history_df = pd.read_csv(HISTORY_PATH)
            st.session_state.history_df = history_df
            
            valid_cols = [c for c in features_needed if c in history_df.columns]
            
            if valid_cols:
                # Media delle audio features
                st.session_state.current_context = history_df[valid_cols].mean().values
                st.session_state.song_count = len(history_df)
                
                # Identifichiamo artista e genere preferiti per il contesto iniziale
                if 'artist' in history_df.columns:
                    st.session_state.top_artist = history_df['artist'].mode()[0]
                if 'genre' in history_df.columns:
                    st.session_state.top_genre = history_df['genre'].mode()[0]
                    
                print(f"✅ Warm Start: Profilo basato su {len(history_df)} brani.")
            else:
                raise ValueError("Nessuna colonna valida trovata.")
        else:
            raise FileNotFoundError("File cronologia non trovato.")

    except Exception as e:
        print(f"ℹ️ Cold Start: {e}")
        st.session_state.history_df = None
        st.session_state.current_context = np.array([0.5] * len(features_needed))
        st.session_state.current_context[8] = 120.0 
        st.session_state.current_context[7] = -8.0  
        st.session_state.song_count = 0
        st.session_state.top_artist = "Unknown"
        st.session_state.top_genre = "Unknown"

    st.session_state.past_track_ids = []
    st.session_state.suggestion_made = False

# --- SIDEBAR: CONTROLLI ---
st.sidebar.header("🎛️ Control Room")

# Mostra i gusti attuali dell'utente
st.sidebar.subheader("Profilo Attuale")
st.sidebar.info(f"🎭 Genere dominante: **{st.session_state.get('top_genre', 'N/A')}**\n\n⭐ Artista top: **{st.session_state.get('top_artist', 'N/A')}**")

if len(st.session_state.oracle.loss_history) > 0:
    st.sidebar.subheader("Apprendimento AI")
    st.sidebar.line_chart(st.session_state.oracle.loss_history)

st.sidebar.markdown("---")
st.sidebar.subheader("Forza il Mood")
curr_en = float(np.clip(st.session_state.current_context[0], 0.0, 1.0))
curr_val = float(np.clip(st.session_state.current_context[1], 0.0, 1.0))
curr_dan = float(np.clip(st.session_state.current_context[2], 0.0, 1.0))

target_energy = st.sidebar.slider("⚡ Energy", 0.0, 1.0, curr_en)
target_valence = st.sidebar.slider("😊 Valence", 0.0, 1.0, curr_val)
target_dance = st.sidebar.slider("💃 Danceability", 0.0, 1.0, curr_dan)

if st.sidebar.button("Applica Modifiche"):
    st.session_state.current_context[0] = target_energy
    st.session_state.current_context[1] = target_valence
    st.session_state.current_context[2] = target_dance
    st.sidebar.success("Mood aggiornato!")

# --- CORE ---
st.subheader("Il prossimo brano per te")

# Predizione AI
predicted_vector = st.session_state.oracle.predict_target(st.session_state.current_context)

if st.button("✨ Genera Suggerimento"):
    try:
        top_song = st.session_state.recommender.get_recommendations(
            predicted_vector, 
            exclude_ids=st.session_state.past_track_ids
        )
        
        st.session_state.last_features = top_song[st.session_state.recommender.features].values
        st.session_state.current_track_name = top_song['track_name']
        st.session_state.current_artist = top_song['artist_name']
        st.session_state.current_track_id = top_song['track_id']
        # Recuperiamo il genere se disponibile nel DB di Kaggle
        st.session_state.current_genre = top_song.get('genres', 'N/A')
        
        st.session_state.past_track_ids.append(str(top_song['track_id']))
        st.session_state.suggestion_made = True
    except Exception as e:
        st.error(f"Errore ricerca: {e}")

# --- DISPLAY ---
if st.session_state.get('suggestion_made'):
    track_id = st.session_state.current_track_id
    
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown(f"### 🎧 {st.session_state.current_track_name}")
        st.markdown(f"**Artista:** {st.session_state.current_artist}")
        st.markdown(f"**Genere suggerito:** `{st.session_state.current_genre}`")
        
        if pd.notna(track_id) and str(track_id).lower() != "nan":
            url = f"https://open.spotify.com/embed/track/{track_id}"
            st.markdown(f'<iframe src="{url}" width="100%" height="152" frameBorder="0" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"></iframe>', unsafe_allow_html=True)
        else:
            st.warning("Anteprima non disponibile.")

    with c2:
        labels = ['Energy', 'Valence', 'Dance', 'Acoust', 'Instr', 'Live', 'Speech', 'Loud', 'Tempo']
        vec_display = predicted_vector.copy()
        vec_display[8] = np.clip(vec_display[8] / 220.0, 0, 1) 
        vec_display[7] = np.clip((vec_display[7] + 60) / 60.0, 0, 1) 
        
        df_r = pd.DataFrame(dict(r=vec_display, theta=labels))
        fig = px.line_polar(df_r, r='r', theta='theta', line_close=True)
        st.plotly_chart(fig)

    st.write("---")
    st.write("Ti piace questo mix?")
    b1, b2 = st.columns([1, 4])
    
    with b1:
        if st.button("👍 Sì"):
            st.session_state.oracle.train_incremental(st.session_state.current_context, st.session_state.last_features)
            st.session_state.song_count += 1
            st.session_state.current_context = calculate_avalanche_context(
                st.session_state.current_context, st.session_state.last_features, st.session_state.song_count
            )
            st.success("L'AI ha imparato la lezione!")
            st.rerun()
            
    with b2:
        if st.button("👎 No"):
            st.warning("Cercherò qualcosa di diverso.")

# --- SEZIONE CRONOLOGIA ---
st.markdown("---")
with st.expander("📜 La tua storia musicale (con Artisti e Generi)"):
    if st.session_state.history_df is not None:
        # Mostriamo anche le nuove colonne artista e genere
        display_cols = ['name', 'artist', 'genre', 'energy', 'valence']
        available_cols = [c for c in display_cols if c in st.session_state.history_df.columns]
        st.dataframe(st.session_state.history_df[available_cols].head(20))
    else:
        st.info("Nessuna cronologia rilevata.")