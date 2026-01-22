import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os  # <--- IMPORTANTE: Serve per gestire i percorsi delle cartelle
from recommender import SongRecommender
from oracle import MusicOracle 
from utils import calculate_avalanche_context

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Billie AI-lish", layout="wide")
st.title("🎵 Billie AI-lish: Music Discovery Agent")

# --- INIZIALIZZAZIONE ---
if 'oracle' not in st.session_state:
    st.session_state.oracle = MusicOracle() 
    
    # 1. Carica il Recommender con il DB reale
    # Nota: Assumiamo che clean_db.csv sia in src (dove c'è lo script). 
    # Se fosse in data, bisognerebbe cambiare path anche qui.
    try:
        st.session_state.recommender = SongRecommender('clean_db.csv')
    except Exception as e:
        st.error(f"⚠️ Errore Database: {e}. Assicurati che 'clean_db.csv' sia nella cartella src.")
        st.stop()

    # 2. WARM START: Carica Storia Utente
    features_needed = st.session_state.recommender.features
    
    # --- CORREZIONE PERCORSO ---
    # Calcoliamo il percorso assoluto per evitare errori:
    # "Dalla cartella dove si trova questo file (src), sali su (..) e vai in 'data'"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    history_path = os.path.join(current_dir, '..', 'data', 'user_history.csv')
    
    print(f"🔍 Cercando la cronologia in: {history_path}")

    try:
        # Leggiamo il file dal percorso corretto
        if os.path.exists(history_path):
            history_df = pd.read_csv(history_path)
        else:
            # Fallback: prova a cercarlo nella cartella corrente se per caso è stato spostato
            history_df = pd.read_csv('user_history.csv')
        
        # Salviamo il dataframe in sessione per visualizzarlo dopo
        st.session_state.history_df = history_df
        
        # Rinomina happiness -> valence se necessario
        if 'happiness' in history_df.columns:
            history_df.rename(columns={'happiness': 'valence'}, inplace=True)
            
        # Calcola media solo colonne valide
        valid_cols = [c for c in features_needed if c in history_df.columns]
        
        if valid_cols:
            initial_context = history_df[valid_cols].mean().values
            
            # Controllo dimensioni
            if len(initial_context) == len(features_needed):
                # Normalizziamo valori "fuori scala" come Tempo e Loudness per l'Input della rete
                st.session_state.current_context = initial_context
                st.session_state.song_count = len(history_df)
                print(f"✅ Warm Start: Profilo basato su {len(history_df)} brani.")
            else:
                raise ValueError("Feature mismatch")
        else:
            raise ValueError("No valid columns")

    except Exception as e:
        print(f"ℹ️ Cold Start (Neutro): Impossibile caricare {history_path}. Errore: {e}")
        st.session_state.history_df = None # Nessuna storia da mostrare
        
        # Default: Valori medi
        st.session_state.current_context = np.array([0.5] * len(features_needed))
        # Fix rapido per Tempo (index 8) e Loudness (index 7) per evitare zeri
        st.session_state.current_context[8] = 120.0 
        st.session_state.current_context[7] = -8.0  
        st.session_state.song_count = 0

    st.session_state.past_track_ids = []
    st.session_state.suggestion_made = False

# --- SIDEBAR: CONTROLLI ---
st.sidebar.header("🎛️ Control Room")

# Grafico Loss
if len(st.session_state.oracle.loss_history) > 0:
    st.sidebar.subheader("Apprendimento AI")
    st.sidebar.line_chart(st.session_state.oracle.loss_history)

# MOOD SLIDERS (Per forzare l'AI)
st.sidebar.markdown("---")
st.sidebar.subheader("Forza il Mood")
st.sidebar.write("Modifica i parametri di input:")

# Recupera valori attuali (con gestione sicura degli indici)
try:
    # Energy=0, Valence=1, Dance=2
    curr_en = float(np.clip(st.session_state.current_context[0], 0.0, 1.0))
    curr_val = float(np.clip(st.session_state.current_context[1], 0.0, 1.0))
    curr_dan = float(np.clip(st.session_state.current_context[2], 0.0, 1.0))
except:
    curr_en, curr_val, curr_dan = 0.5, 0.5, 0.5

target_energy = st.sidebar.slider("⚡ Energy", 0.0, 1.0, curr_en)
target_valence = st.sidebar.slider("😊 Valence (Happy)", 0.0, 1.0, curr_val)
target_dance = st.sidebar.slider("💃 Danceability", 0.0, 1.0, curr_dan)

if st.sidebar.button("Applica Modifiche"):
    new_ctx = st.session_state.current_context.copy()
    new_ctx[0] = target_energy
    new_ctx[1] = target_valence
    new_ctx[2] = target_dance
    st.session_state.current_context = new_ctx
    st.sidebar.success("Mood aggiornato!")

# --- CORE ---
st.subheader("Il prossimo brano per te")

# 1. Predizione AI
predicted_vector = st.session_state.oracle.predict_target(st.session_state.current_context)

if st.button("✨ Genera Suggerimento"):
    try:
        # 2. Ricerca KNN
        top_song = st.session_state.recommender.get_recommendations(
            predicted_vector, 
            exclude_ids=st.session_state.past_track_ids
        )
        
        # Salvataggio stato
        st.session_state.last_features = top_song[st.session_state.recommender.features].values
        st.session_state.current_track_name = top_song['track_name']
        st.session_state.current_artist = top_song['artist_name']
        st.session_state.current_track_id = top_song['track_id']
        
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
        st.markdown(f"**{st.session_state.current_artist}**")
        
        # Player
        if track_id and str(track_id).lower() != "nan":
            url = f"https://open.spotify.com/embed/track/{track_id}?utm_source=generator"
            st.markdown(f'<iframe src="{url}" width="100%" height="152" frameBorder="0" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"></iframe>', unsafe_allow_html=True)
        else:
            st.warning("Anteprima non disponibile (ID mancante).")

    with c2:
        # Radar Chart
        labels = ['Energy', 'Valence', 'Dance', 'Acoust', 'Instr', 'Live', 'Speech', 'Loud', 'Tempo']
        # Normalizziamo il vettore per la visualizzazione
        vec_display = predicted_vector.copy()
        vec_display[8] = min(vec_display[8] / 200.0, 1.0) # Tempo Scale
        vec_display[7] = (vec_display[7] + 60) / 60.0    # Loudness Scale
        
        df_r = pd.DataFrame(dict(r=vec_display, theta=labels))
        fig = px.line_polar(df_r, r='r', theta='theta', line_close=True)
        st.plotly_chart(fig)

    # --- FEEDBACK ---
    st.write("---")
    st.write("Ti piace?")
    b1, b2 = st.columns([1, 4])
    
    with b1:
        if st.button("👍 Sì"):
            st.session_state.oracle.train_incremental(
                st.session_state.current_context, 
                st.session_state.last_features
            )
            st.session_state.song_count += 1
            st.session_state.current_context = calculate_avalanche_context(
                st.session_state.current_context,
                st.session_state.last_features,
                st.session_state.song_count
            )
            st.success("AI Addestrata!")
            st.rerun()
            
    with b2:
        if st.button("👎 No"):
            st.warning("Skippato.")

# --- SEZIONE CRONOLOGIA (NUOVA) ---
st.markdown("---")
with st.expander("📜 Visualizza la tua Cronologia Analizzata (Ultimi 50 brani)"):
    if st.session_state.history_df is not None:
        # Selezioniamo le colonne più belle da vedere
        cols_to_show = ['name', 'artist', 'energy', 'valence', 'danceability', 'tempo']
        # Filtriamo solo le colonne che esistono davvero nel CSV
        visible_cols = [c for c in cols_to_show if c in st.session_state.history_df.columns]
        
        st.dataframe(st.session_state.history_df[visible_cols])
        st.caption(f"Totale brani analizzati: {len(st.session_state.history_df)}")
    else:
        st.info(f"Nessuna cronologia trovata in {history_path}. Controlla che il file esista.")