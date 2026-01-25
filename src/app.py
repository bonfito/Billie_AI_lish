import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
from recommender import SongRecommender
from oracle import MusicOracle 
from utils import calculate_avalanche_context
# Assicurati di avere anche get_track_cover se lo hai implementato, altrimenti toglilo dall'import
from spotify_client import add_track_to_playlist 

# --- CONFIGURAZIONE PERCORSI ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Percorsi relativi alla cartella data
HISTORY_PATH = os.path.normpath(os.path.join(CURRENT_DIR, '..', 'data', 'user_history.csv'))
BLACKLIST_PATH = os.path.normpath(os.path.join(CURRENT_DIR, '..', 'data', 'blacklist.txt'))

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Billie AI-lish", layout="wide", page_icon="🎵")

# CSS Custom per un look moderno
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 12px; font-weight: bold; }
    .big-font { font-size: 28px !important; font-weight: bold; color: #1DB954; }
    .artist-font { font-size: 20px !important; font-weight: 500; }
    .reason-tag { 
        background-color: #f0f2f6; 
        padding: 5px 12px; 
        border-radius: 15px; 
        font-size: 0.9em; 
        color: #333; 
        border: 1px solid #d1d5db;
        display: inline-block;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎵 Billie AI-lish: Music Discovery Agent")

# --- 1. INIZIALIZZAZIONE SISTEMA ---
if 'oracle' not in st.session_state:
    st.session_state.oracle = MusicOracle() 
    
    # Caricamento Recommender
    try:
        with st.spinner("Caricamento del cervello musicale (2.8M brani)..."):
            # Chiamata senza argomenti (il recommender si gestisce i path da solo)
            st.session_state.recommender = SongRecommender()
            st.success(f"✅ Motore di raccomandazione attivo.")
            
    except Exception as e:
        st.error(f"⚠️ Errore Critico Database: {e}")
        st.stop()

    # Inizializzazione variabili sessione
    st.session_state.past_track_ids = []
    st.session_state.suggestion_made = False
    st.session_state.current_track = None
    st.session_state.predicted_vector = None

# --- 2. CARICAMENTO STORIA UTENTE ---
if 'history_df' not in st.session_state:
    try:
        if os.path.exists(HISTORY_PATH):
            history_df = pd.read_csv(HISTORY_PATH)
            st.session_state.history_df = history_df
            
            features_needed = st.session_state.recommender.audio_cols
            valid_cols = [c for c in features_needed if c in history_df.columns]
            
            if valid_cols:
                # Calcolo contesto iniziale
                st.session_state.current_context = history_df[valid_cols].mean().values
                st.session_state.song_count = len(history_df)
                
                # Top Artist/Genre
                st.session_state.top_artist = history_df['artist'].mode()[0] if 'artist' in history_df else "Unknown"
                st.session_state.top_genre = history_df['genre'].mode()[0] if 'genre' in history_df else "Unknown"
                
                print(f"✅ Warm Start: Profilo basato su {len(history_df)} brani.")
            else:
                raise ValueError("Nessuna colonna audio valida trovata.")
        else:
            raise FileNotFoundError("File cronologia non trovato.")

    except Exception as e:
        # COLD START
        st.session_state.history_df = None
        default_ctx = np.array([0.5] * 9)
        st.session_state.current_context = default_ctx
        st.session_state.song_count = 0
        st.session_state.top_artist = "N/A"
        st.session_state.top_genre = "N/A"

# --- SIDEBAR: CONTROL ROOM ---
st.sidebar.header("🎛️ Control Room")

# Info Profilo
st.sidebar.subheader("Profilo Attuale")
st.sidebar.info(f"🎭 Genere dominante: **{st.session_state.get('top_genre', 'N/A')}**\n\n⭐ Artista top: **{st.session_state.get('top_artist', 'N/A')}**")

# Grafico Loss
if len(st.session_state.oracle.loss_history) > 0:
    st.sidebar.subheader("Apprendimento AI")
    st.sidebar.line_chart(st.session_state.oracle.loss_history)
    st.sidebar.caption("L'errore diminuisce man mano che impariamo i tuoi gusti.")

st.sidebar.markdown("---")

# Sliders Mood
st.sidebar.subheader("🎚️ Equalizzatore Mood")
if st.session_state.current_context is not None:
    curr_en = float(np.clip(st.session_state.current_context[0], 0.0, 1.0))
    curr_val = float(np.clip(st.session_state.current_context[1], 0.0, 1.0))
    curr_dan = float(np.clip(st.session_state.current_context[2], 0.0, 1.0))

    target_energy = st.sidebar.slider("⚡ Energy", 0.0, 1.0, curr_en)
    target_valence = st.sidebar.slider("😊 Valence (Felicità)", 0.0, 1.0, curr_val)
    target_dance = st.sidebar.slider("💃 Danceability", 0.0, 1.0, curr_dan)

    if st.sidebar.button("Applica Modifiche Manuali"):
        st.session_state.current_context[0] = target_energy
        st.session_state.current_context[1] = target_valence
        st.session_state.current_context[2] = target_dance
        st.sidebar.success("Mood aggiornato! Genera un nuovo suggerimento.")

# --- MAIN CONTENT ---
col_main, col_radar = st.columns([1.5, 1])

with col_main:
    st.subheader("🧬 Il prossimo brano per te")
    
    # 1. GENERAZIONE
    if st.button("✨ Genera Nuova Visione", type="primary"):
        if st.session_state.history_df is not None:
            try:
                # --- CHIAMATA AL RECOMMENDER ---
                recs_df, pred_vector = st.session_state.recommender.recommend(
                    st.session_state.history_df, 
                    k=20
                )
                
                # Prendiamo il Primo Risultato
                best_song = recs_df.iloc[0]
                
                # Salviamo nello stato
                st.session_state.current_track = best_song.to_dict()
                st.session_state.predicted_vector = pred_vector.flatten()
                st.session_state.suggestion_made = True
                
                # Aggiorniamo la lista degli ID esclusi temporaneamente
                st.session_state.past_track_ids.append(str(best_song['id']))
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Errore durante la generazione: {e}")
        else:
            st.error("Nessuna cronologia disponibile. Esegui 'fetch_history.py'.")

    # 2. VISUALIZZAZIONE RISULTATO
    if st.session_state.suggestion_made and st.session_state.current_track:
        track = st.session_state.current_track
        
        st.write("---")
        # Titolo
        st.markdown(f"<div class='big-font'>{track['name']}</div>", unsafe_allow_html=True)
        # Artista
        st.markdown(f"<div class='artist-font'>🎤 {track['artist']}</div>", unsafe_allow_html=True)
        
        # Reason Tag
        reason = track.get('reason_text', 'Suggerimento AI')
        st.markdown(f"<div class='reason-tag'>💡 {reason}</div>", unsafe_allow_html=True)
        
        # Info Aggiuntive
        c_info1, c_info2 = st.columns(2)
        genre_display = track.get('genres', st.session_state.get('current_genre', 'N/A'))
        if genre_display == 'unknown': genre_display = "Genre Fluid"
        
        c_info1.caption(f"📅 Anno: {int(track.get('year', 0))}")
        c_info2.caption(f"🎹 Genere: {genre_display}")

        # Spotify Player Ufficiale
        tid = track.get('id')
        if pd.notna(tid):
            url = f"https://open.spotify.com/embed/track/{tid}?utm_source=generator"
            st.markdown(f'<iframe src="{url}" width="100%" height="152" frameBorder="0" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>', unsafe_allow_html=True)
        else:
            st.warning("Anteprima audio non disponibile (ID mancante).")

        # 3. FEEDBACK LOOP
        st.write("### Ti piace questo mix?")
        b1, b2 = st.columns([1, 4])
        
        with b1:
            # FIX: Aggiunto key univoca "btn_like"
            if st.button("👍 Sì", key="btn_like"):
                # Recuperiamo le feature del brano corrente
                cols = st.session_state.recommender.audio_cols
                last_features = np.array([track[c] for c in cols])
                
                # A. Addestramento Oracle
                st.session_state.oracle.train_incremental(
                    st.session_state.current_context, 
                    last_features
                )
                
                # B. Aggiornamento Contesto
                st.session_state.song_count += 1
                st.session_state.current_context = calculate_avalanche_context(
                    st.session_state.current_context, 
                    last_features, 
                    st.session_state.song_count
                )
                
                # C. Salvataggio su Playlist Spotify
                tid = track.get('id')
                if tid:
                    success, msg = add_track_to_playlist(tid)
                    if success:
                        st.toast(f"Aggiunta a '{msg}'!", icon="✅")
                    else:
                        st.error(f"Errore Spotify: {msg}")
                
                st.toast("AI Aggiornata! Il DNA si è evoluto.", icon="🧬")
                # Non facciamo rerun qui per lasciare l'utente godersi la canzone
                
        with b2:
            # FIX: Aggiunto key univoca "btn_dislike"
            if st.button("👎 No", key="btn_dislike"):
                # Salviamo l'ID in un file di testo 'blacklist' per non riproporla mai più
                try:
                    with open(BLACKLIST_PATH, "a") as f:
                        f.write(f"{track['id']}\n")
                except Exception as e:
                    print(f"Errore scrittura blacklist: {e}")

                st.session_state.past_track_ids.append(str(track['id']))
                st.toast("Ricevuto. Mai più questa canzone.", icon="🚫")
                st.rerun() # Ricarica immediata per togliere il brano dalla vista

# --- RADAR CHART (Colonna Destra) ---
with col_radar:
    if st.session_state.suggestion_made and st.session_state.predicted_vector is not None:
        st.markdown("### 🎯 Target DNA")
        
        # Etichette standard
        labels = ['Energy', 'Valence', 'Dance', 'Tempo', 'Loud', 'Speech', 'Acoust', 'Instr', 'Live']
        
        # Preparazione dati
        vec_display = st.session_state.predicted_vector.flatten().copy()
        
        if len(vec_display) >= 9:
            vec_display = np.clip(vec_display, 0, 1)

        df_r = pd.DataFrame(dict(r=vec_display[:9], theta=labels))
        
        fig = px.line_polar(df_r, r='r', theta='theta', line_close=True, range_r=[0, 1])
        fig.update_layout(margin=dict(t=30, b=30, l=30, r=30))
        st.plotly_chart(fig, use_container_width=True)

# --- SEZIONE CRONOLOGIA (Bottom) ---
st.write("---")
with st.expander("📜 La tua storia musicale"):
    if st.session_state.history_df is not None:
        display_cols = ['name', 'artist', 'popularity', 'year']
        available = [c for c in display_cols if c in st.session_state.history_df.columns]
        st.dataframe(st.session_state.history_df[available].head(15), use_container_width=True)
    else:
        st.info("Nessuna cronologia rilevata.")