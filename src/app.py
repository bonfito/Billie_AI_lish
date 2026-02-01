import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import sys
import time
import joblib
from dotenv import load_dotenv

# Import dai moduli locali
from recommender import SongRecommender
from oracle import MusicOracle 
from utils import calculate_avalanche_context
from spotify_client import add_track_to_playlist, get_track_details
from fetch_userhistory import fetch_history

# --- CONFIGURAZIONE PERCORSI ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = CURRENT_DIR
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

DATA_DIR = os.path.normpath(os.path.join(CURRENT_DIR, '..', 'data'))
HISTORY_PATH = os.path.join(DATA_DIR, 'user_history.csv')
PLAYLIST_SAVED_PATH = os.path.join(DATA_DIR, 'playlist_saved.csv')
LIKED_PATH = os.path.join(DATA_DIR, 'liked.csv')
DISLIKED_PATH = os.path.join(DATA_DIR, 'disliked.csv')
SCALER_PATH = os.path.join(DATA_DIR, 'scaler.save')
ORACLE_PATH = os.path.join(DATA_DIR, 'oracle.pkl')

# Carica variabili ambiente
load_dotenv()

# --- CARICAMENTO SCALER ---
try:
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    scaler = None

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(
    page_title="Billie AI-lish", 
    layout="centered", 
    page_icon=None,
    initial_sidebar_state="expanded"
)

# --- CSS ---
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {
        visibility: visible !important;
        background: transparent !important;
    }

    /* 1. Layout Principale */
    .block-container {
        max-width: 900px !important; 
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        margin: 0 auto;
        padding-top: 2rem;
    }

    section[data-testid="stMain"] [data-testid="stVerticalBlock"] > div {
        width: 100% !important;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    /* 2. Tipografia */
    .main-title { font-size: 3.5rem; font-weight: 900; color: #FFFFFF; text-transform: uppercase; margin-bottom: 0; }
    .subtitle { font-size: 1rem; letter-spacing: 5px; color: #1DB954; text-transform: uppercase; margin-bottom: 3rem; }
    
    .track-name { font-size: 3rem; font-weight: 800; margin-top: 1.5rem; line-height: 1.1; color: #fff; }
    .artist-name { font-size: 1.6rem; font-weight: 400; color: #1DB954; margin-bottom: 0.5rem; }
    .meta-tag { font-size: 0.9rem; color: #555; letter-spacing: 2px; margin-bottom: 2rem; text-transform: uppercase; }

    /* 3. Spotify Player */
    .spotify-container {
        width: 100%;
        min-height: 352px;
        display: flex;
        justify-content: center;
        margin: 0 auto;
    }
    iframe {
        border-radius: 20px;
        box-shadow: 0 20px 50px rgba(0,0,0,0.6);
    }
    
    /* 4. Lista Feature a Destra (Track DNA) */
    .feature-list {
        text-align: left;
        background-color: #111;
        padding: 15px;
        margin-left:-200px;
        border-radius: 15px;
        border: 1px solid #333;
        height: 352px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .feature-item {
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
        border-bottom: 1px solid #222;
        padding-bottom: 2px;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
    }
    .feat-label { color: #888; font-weight: bold; }
    .feat-val { color: #1DB954; }

    /* 5. HISTORY TABLE */
    .history-container {
        width: 100% !important;
        max-width: 850px; 
        height: 250px !important; 
        overflow-y: auto; 
        margin: 20px auto; 
        padding: 0 15px;
        border-top: 1px solid #333;
        border-bottom: 1px solid #333;
        display: block;
    }
    
    .history-container::-webkit-scrollbar { width: 6px; }
    .history-container::-webkit-scrollbar-thumb { background: #1DB954; border-radius: 10px; }

    .history-table { 
        width: 100%; 
        border-collapse: collapse; 
        font-family: 'Courier New', monospace;
        margin: 0 auto;
    }

    .history-table td {
        padding: 12px 10px;
        border-bottom: 1px solid #1a1a1a;
        font-size: 0.85rem;
        color: #888;
        vertical-align: middle;
        text-align: left; 
    }

    .track-number { width: 50px; color: #444; font-weight: bold; }
    .track-title-cell { color: #eee; letter-spacing: -0.5px; }
    .history-row-artist { color: #1DB954; font-weight: 600; opacity: 0.8; }
    .history-table tr:hover td { background-color: #111; color: #fff; }

    /* 6. Bottoni Main */
    section[data-testid="stMain"] .stButton > button {
        width: 100% !important;
        border-radius: 50px;
        border: 2px solid #333;
        background: transparent;
        color: white;
        font-weight: bold;
        padding: 0.6rem;
        transition: 0.3s;
    }
    section[data-testid="stMain"] .stButton > button:hover { border-color: #1DB954; color: #1DB954; transform: scale(1.02); }
    </style>
""", unsafe_allow_html=True)

# --- 1. INIZIALIZZAZIONE ---
if 'oracle' not in st.session_state:
    with st.spinner("Sintonizzando Billie AI-lish..."):

        try:
            if os.path.exists(ORACLE_PATH):
                st.session_state.oracle = joblib.load(ORACLE_PATH)
            else:
                st.session_state.oracle = MusicOracle()
        
        except ModuleNotFoundError:
            st.sidebar.warning("Oracle rigenerato, cambio struttura")
            st.session_state.oracle = MusicOracle()
        except Exception as e:
            st.sidebar.warning(f"Errore caricamento Oracle: {e}")
            st.session_state.oracle = MusicOracle()


        try:
            st.session_state.recommender = SongRecommender()
        except Exception as e:
            st.error(f"System Error: {e}")
            st.stop()
    st.session_state.past_track_ids = []
    st.session_state.suggestion_made = False
    st.session_state.current_track = None
    st.session_state.predicted_vector = None
    
    # --- STATO CODA E VIBE ---
    st.session_state.recs_queue = pd.DataFrame()
    st.session_state.vibe_history = [50] # Parte da 50 (Neutro)
    
    # --- NUOVO: BLACKLIST TEMPORANEA (SESSIONE BROWSER) ---
    # Qui finiscono le canzoni che metti Like/Dislike per non rivederle subito
    st.session_state.session_blacklist = []

# --- FUNZIONE GENERAZIONE (BUFFERIZZATA + FILTRO BLACKLIST) ---
def generate_new_recommendation(manual_target=None):
    if st.session_state.history_df is not None:
        try:
            # 1. Se c'è un target manuale (Slider), ricalcoliamo tutto e resettiamo la coda
            if manual_target is not None:
                st.session_state.recs_queue = pd.DataFrame() # Reset coda
                recs_df, pred_vector = st.session_state.recommender.recommend(
                    st.session_state.history_df, 
                    k=30, 
                    target_features=manual_target,
                    # Passiamo la blacklist della sessione al Recommender
                    session_blacklist=st.session_state.session_blacklist 
                )
            
            # 2. Se la coda è vuota, calcoliamo un nuovo batch
            elif st.session_state.recs_queue.empty:
                recs_df, pred_vector = st.session_state.recommender.recommend(
                    st.session_state.history_df, 
                    k=30,
                    # Passiamo la blacklist della sessione al Recommender
                    session_blacklist=st.session_state.session_blacklist
                )
            
            # 3. Se la coda ha canzoni, usiamo quelle (ISTANTANEO)
            else:
                recs_df = st.session_state.recs_queue
                # Manteniamo il vettore target precedente
                pred_vector = st.session_state.predicted_vector if st.session_state.predicted_vector is not None else np.zeros(9)

            # --- CONTROLLO ANTI-CRASH ---
            if recs_df is None or recs_df.empty:
                st.warning("Nessuna canzone trovata. I filtri potrebbero essere troppo stretti.")
                return False
            
            # Estrai la prima canzone
            best_song = recs_df.iloc[0]

            # Aggiorna la coda: togli la canzone scelta, tieni le altre 29
            st.session_state.recs_queue = recs_df.iloc[1:].reset_index(drop=True)

            # Aggiorna stato
            st.session_state.current_track = best_song.to_dict()
            if manual_target is not None or st.session_state.predicted_vector is None:
                st.session_state.predicted_vector = pred_vector.flatten()
                
            st.session_state.suggestion_made = True
            
            if 'past_track_ids' not in st.session_state: st.session_state.past_track_ids = []
            st.session_state.past_track_ids.append(str(best_song['id']))
            
            return True
        except Exception as e:
            st.error(f"Errore generazione: {e}")
            return False
    return False

# --- FUNZIONE SALVATAGGIO FEEDBACK (LIKE / DISLIKE) ---
def append_feedback_csv(csv_path, track_dict, real_g=None, real_p=None):
    """Salva una traccia su un CSV dedicato (liked/disliked)"""
    try:
        cols = st.session_state.recommender.audio_cols
        row = {
            'id': track_dict.get('id'),
            'name': track_dict.get('name'),
            'artist': track_dict.get('artist'),
            'genres': real_g if real_g is not None else track_dict.get('genres', 'unknown'),
            'popularity': real_p if real_p is not None else track_dict.get('popularity', None),
            'year': track_dict.get('year'),
            **{k: track_dict.get(k, 0) for k in cols}
        }
        df_new = pd.DataFrame([row])
        df_new.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
    except Exception as e:
        st.warning(f"Impossibile salvare feedback su {os.path.basename(csv_path)}: {e}")

# --- FUNZIONE AGGIORNAMENTO VIBE ---
def update_vibe(points):
    """Aggiorna il grafico della vibe (0-100)"""
    current_score = st.session_state.vibe_history[-1]
    new_score = np.clip(current_score + points, 0, 100)
    st.session_state.vibe_history.append(new_score)
    # Teniamo solo gli ultimi 50 punti per il grafico
    if len(st.session_state.vibe_history) > 50:
        st.session_state.vibe_history.pop(0)

# --- 2. CARICAMENTO STORIA ---

@st.cache_data(show_spinner=False)
def load_data(history_path: str) -> pd.DataFrame:
    """Carica lo storico da CSV (se esiste)."""
    if os.path.exists(history_path):
        return pd.read_csv(history_path)
    return pd.DataFrame()

# --- AUTO-FETCH HISTORY ALL'AVVIO (UNA VOLTA PER SESSIONE) ---
if 'history_fetched' not in st.session_state:
    try:
        fetch_history()
    except Exception:
        # Non blocchiamo l'app se il fetch fallisce
        pass
    st.session_state.history_fetched = True
    # Se il fetch ha scritto un nuovo CSV, resettiamo la cache
    try:
        load_data.clear()
    except Exception:
        pass

if 'history_df' not in st.session_state:
    try:
        history_df = load_data(HISTORY_PATH)
        if history_df is not None and not history_df.empty:
            st.session_state.history_df = history_df
            features = st.session_state.recommender.audio_cols
            valid = [c for c in features if c in history_df.columns]
            if valid:
                st.session_state.current_context = history_df[valid].mean().values
                st.session_state.song_count = len(history_df)

                genre_col = 'genres' if 'genres' in history_df.columns else 'genre'
                if genre_col in history_df.columns:
                    v_gen = history_df[~history_df[genre_col].isin(['unknown', 'nan'])][genre_col]
                    st.session_state.top_genre = v_gen.mode()[0].title() if not v_gen.empty else "N/A"
                v_art = history_df[~history_df['artist'].isin(['unknown', 'nan'])]['artist']
                st.session_state.top_artist = v_art.mode()[0] if not v_art.empty else "N/A"
        else:
            raise FileNotFoundError()
    except Exception:
        st.session_state.history_df = None
        st.session_state.current_context = np.array([0.5] * 9)
        st.session_state.song_count = 0
        st.session_state.top_artist = "-"
        st.session_state.top_genre = "-"

# --- HEADER ---
st.markdown("<div class='main-title'>BILLIE AI-LISH</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Artificial Music Agent</div>", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.header("CONTROL ROOM")
if st.session_state.get('oracle') and st.session_state.oracle.loss_history:
    n_trained = len(st.session_state.oracle.loss_history)
    st.sidebar.success(f"Oracle: {n_trained} interazioni")
    
    # Mostra andamento Loss (deve scendere!)
    if n_trained > 1:
        first_loss = st.session_state.oracle.loss_history[0]
        last_loss = st.session_state.oracle.loss_history[-1]
        improvement = ((first_loss - last_loss) / first_loss) * 100
        
        st.sidebar.caption(f"Loss iniziale: {first_loss:.4f}")
        st.sidebar.caption(f"Loss attuale: {last_loss:.4f}")
        
        if improvement > 0:
            st.sidebar.caption(f"Miglioramento: {improvement:.1f}%")
        else:
            st.sidebar.caption(f"Loss in crescita (normale all'inizio)")
st.sidebar.caption(f"Genre: {st.session_state.get('top_genre', '-')} | Artist: {st.session_state.get('top_artist', '-')}")

# --- INDICATORE CODA ---
q_len = len(st.session_state.recs_queue) if 'recs_queue' in st.session_state else 0
if q_len > 0:
    st.sidebar.success(f"Coda Veloce Attiva: {q_len} brani pronti")
else:
    st.sidebar.info("Coda vuota (Il prossimo click calcolerà un nuovo batch)")

st.sidebar.markdown("---")

if st.sidebar.button("Aggiorna Cronologia"):
    with st.spinner("Scaricamento dati..."):
        try:
            fetch_history()
            if 'history_df' in st.session_state: del st.session_state['history_df']
            load_data.clear()
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Errore: {e}")

# --- GENERAZIONE ---
c1, col_gen, c3 = st.columns([1, 2, 1])
with col_gen:
    btn_label = "GENERA PROSSIMA (Instant)" if q_len > 0 else "AVVIA SESSIONE (Load)"
    
    if st.button(btn_label, type="primary", key="main_gen"):
        if q_len == 0:
            with st.spinner("L'AI sta scansionando il database..."):
                if generate_new_recommendation():
                    st.rerun()
        else:
            generate_new_recommendation()
            st.rerun()

# --- DISPLAY CANZONE ---
if st.session_state.suggestion_made and st.session_state.current_track:
    track = st.session_state.current_track
    tid = track.get('id')
    
    col_player, col_stats = st.columns([2, 1]) 
    
    with col_player:
        if pd.notna(tid):
            url = f"https://open.spotify.com/embed/track/{tid}?utm_source=generator&theme=0"
            st.markdown(f'<div class="spotify-container"><iframe src="{url}" width="100%" height="352" frameBorder="0" allow="autoplay; encrypted-media; fullscreen; picture-in-picture"></iframe></div>', unsafe_allow_html=True)
        else:
            st.warning("Anteprima non disponibile per questa traccia.")

    with col_stats:
        audio_cols = ['energy', 'valence', 'danceability', 'tempo', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness']
        norm_vector = np.array([track.get(c, 0) for c in audio_cols]).reshape(1, -1)
        real_data_map = {}
        if scaler:
            try:
                real_vector = scaler.inverse_transform(norm_vector)[0]
                real_data_map = dict(zip(audio_cols, real_vector))
            except: pass 
        
        display_feats = audio_cols 
        html_stats = "<div class='feature-list'><div style='color:#fff; font-weight:900; margin-bottom:10px; text-transform:uppercase; letter-spacing:2px; font-size:0.9rem;'>Track DNA</div>"
        
        for f in display_feats:
            val_norm = track.get(f, 0)
            if f == 'tempo':
                bpm = real_data_map.get(f, val_norm * 160 + 40)
                val_str = f"{int(bpm)} BPM"
            elif f == 'loudness':
                db = real_data_map.get(f, val_norm * 60 - 60)
                val_str = f"{db:.1f} dB"
            else:
                val_str = f"{int(val_norm * 100)}"
            html_stats += f"<div class='feature-item'><span class='feat-label'>{f.capitalize()}</span><span class='feat-val'>{val_str}</span></div>"
        html_stats += "</div>"
        st.markdown(html_stats, unsafe_allow_html=True)

    st.markdown(f"<div class='track-name'>{track['name']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='artist-name'>{track['artist']}</div>", unsafe_allow_html=True)
    g = str(track.get('genres', 'unknown')).title()
    y = int(track.get('year', 0))
    st.markdown(f"<div class='meta-tag'>{y} • {g}</div>", unsafe_allow_html=True)

    # --- BOTTONI (Like, Dislike, Save) ---
    c_dislike, c_like, c_save = st.columns([1, 1, 2])
    
    #  DISLIKE (Scende Vibe, Blacklist Sessione, Next)
    with c_dislike:
        if st.button("DISLIKE", key="btn_dislike"):
            update_vibe(-10) # Scende Vibe

            # Salva su CSV dedicato (DISLIKED)
            real_g, real_p = get_track_details(track['id'])
            append_feedback_csv(DISLIKED_PATH, track, real_g, real_p)

            # Anche se non piace, non vogliamo rivederla subito nella stessa sessione
            st.session_state.session_blacklist.append(track['id'])

            st.session_state.past_track_ids.append(str(track['id']))
            generate_new_recommendation()
            st.rerun()

    # LIKE (Sale Vibe, Allena AI, Blacklist Sessione, Next)
    with c_like:
        if st.button("LIKE", key="btn_like"):
            update_vibe(+10) # Sale Vibe

            # Salva su CSV dedicato (LIKED) - NON in user_history.csv
            real_g, real_p = get_track_details(track['id'])
            append_feedback_csv(LIKED_PATH, track, real_g, real_p)

            # 1. Allenamento Oracle
            cols = st.session_state.recommender.audio_cols
            feats = np.array([track[k] for k in cols])
            if st.session_state.oracle:

                print("\n" + "="*60)
                print(f"LIKE - {track['name']} by {track['artist']}")
                print("="*60)

                #predizione priam del training
                prediction_before = st.session_state.oracle.predict_target(st.session_state.current_context)
                loss_count_before = len(st.session_state.oracle.loss_history)

                print(f"Stato PRIMA DEL TRAINING:")
                print(f"   • Training iterations: {loss_count_before}")

                if loss_count_before > 0:
                    print(f"   • Last loss: {st.session_state.oracle.loss_history[-1]:.6f}")

                #TRAINING
                st.session_state.oracle.train_incremental(st.session_state.current_context, feats)

                loss_count_after = len(st.session_state.oracle.loss_history)
                current_loss = st.session_state.oracle.loss_history[-1]

                print(f"Stato dopo il training: ")
                print(f"   • Training iterations: {loss_count_after}")
                print(f"   • Current loss: {current_loss:.6f}")

                # Confronto predizione
                prediction_after = st.session_state.oracle.predict_target(st.session_state.current_context)
                diff = np.abs(prediction_after - prediction_before).mean()
                print(f"   • Cambio medio predizione: {diff:.6f}")
            
                # Confronto con target reale
                error = np.abs(feats - prediction_after).mean()
                print(f"   • Errore medio vs target: {error:.6f}")
            
                print("\n Predizione vs Realtà (prime 5 features):")
                feature_names = ['Energy', 'Valence', 'Dance', 'Tempo', 'Loud']
                for i, name in enumerate(feature_names):
                    print(f"   • {name:12}: Pred={prediction_after[i]:.3f} | Real={feats[i]:.3f} | Diff={abs(prediction_after[i]-feats[i]):.3f}")
            
                # Andamento Loss
                if loss_count_after > 1:
                    loss_trend = st.session_state.oracle.loss_history[-5:]  # Ultimi 5
                    print(f"\n Loss trend (ultimi 5):")
                    for idx, loss in enumerate(loss_trend, start=1):
                        print(f"   {idx}. {loss:.6f}")
            
                print("="*60 + "\n")

                #SALVATAGGIO
                try:
                    joblib.dump(st.session_state.oracle, ORACLE_PATH)
                except Exception as e:
                    st.warning(f"Oracle non salvato: {e}")

            # 2. Aggiorna Contesto
            st.session_state.current_context = calculate_avalanche_context(st.session_state.current_context, feats, st.session_state.song_count)

            # Anche se piace, non vogliamo risentirla subito nella stessa sessione
            st.session_state.session_blacklist.append(track['id'])

            st.session_state.past_track_ids.append(str(track['id']))
            generate_new_recommendation() 
            st.rerun()

    #  SAVE (Sale Vibe Molto, Allena AI, Salva File, Next)
    with c_save:
        if st.button("SALVA IN LIBRARY", key="btn_save"):
            with st.status("Salvataggio...", expanded=False) as status:
                update_vibe(+20) # Vibe sale molto
                
                real_g, real_p = get_track_details(track['id'])
                cols = st.session_state.recommender.audio_cols
                feats = np.array([track[k] for k in cols])
                
                if st.session_state.oracle:

                    print("\n" + "="*60)
                    print(f"SAVE - {track['name']} by {track['artist']}")
                    print("="*60)   

                    loss_count_before = len(st.session_state.oracle.loss_history)
                    print(f"Training iterations prima: {loss_count_before}")


                    st.session_state.oracle.train_incremental(st.session_state.current_context, feats)

                    loss_count_after = len(st.session_state.oracle.loss_history)
                    current_loss = st.session_state.oracle.loss_history[-1]
                    print(f"Training iterations dopo: {loss_count_after}")
                    print(f"Current loss: {current_loss:.6f}")
                    print("="*60 + "\n")

                    #salvataggio oracle su disco
                    try:
                        joblib.dump(st.session_state.oracle, ORACLE_PATH)
                    except Exception as e:
                        pass
                
                st.session_state.song_count += 1
                st.session_state.current_context = calculate_avalanche_context(st.session_state.current_context, feats, st.session_state.song_count)
                
                if tid: add_track_to_playlist(tid)
                
                # --- MODIFICA RICHIESTA: AGGIUNGI A BLACKLIST SESSIONE ---
                st.session_state.session_blacklist.append(track['id'])
                
                # Salva su CSV dedicato (playlist_saved.csv) e NON in user_history.csv
                new_row = {'id': track['id'], 'name': track['name'], 'artist': track['artist'], 'genres': real_g, 'popularity': real_p, 'year': track.get('year'), **{k: track[k] for k in cols}}
                df_new = pd.DataFrame([new_row])
                df_new.to_csv(PLAYLIST_SAVED_PATH, mode='a', header=not os.path.exists(PLAYLIST_SAVED_PATH), index=False)
                
                generate_new_recommendation()
                status.update(label="Salvato!", state="complete")
            st.rerun()

st.markdown("---")

# --- GRAFICO VIBE (SATISFACTION) ---
st.markdown("<div style='letter-spacing: 2px; font-weight: 900; color: #444; margin-top:20px; font-size: 0.8rem;'>SESSION VIBE</div>", unsafe_allow_html=True)
vibe_data = pd.DataFrame(st.session_state.vibe_history, columns=['Vibe'])
st.line_chart(vibe_data, height=150, color='#1DB954')

# --- HISTORY & RADAR ---
col_hist, col_radar = st.columns([1, 1])

with col_hist:
    st.markdown("<div style='letter-spacing: 5px; font-weight: 900; color: #444; margin-top:20px; font-size: 0.7rem;'>LATEST DISCOVERIES</div>", unsafe_allow_html=True)
    if st.session_state.history_df is not None:
        recent = st.session_state.history_df[['name', 'artist']].tail(50).iloc[::-1].reset_index(drop=True)
        total = len(recent)
        html_h = "<div class='history-container'><table class='history-table'>"
        for i, r in recent.iterrows():
            num = str(total - i).zfill(2)
            html_h += f"<tr><td class='track-number'>{num}</td><td class='track-title-cell'>{r['name']} <span class='history-row-artist'> // {r['artist']}</span></td></tr>"
        html_h += "</table></div>"
        st.markdown(html_h, unsafe_allow_html=True)

with col_radar:
    st.markdown("<div style='letter-spacing: 2px; font-weight: 900; color: #444; margin-top:20px; font-size: 0.8rem;'>TARGET DNA</div>", unsafe_allow_html=True)
    vector_to_plot = st.session_state.predicted_vector
    if vector_to_plot is None and st.session_state.current_context is not None:
        vector_to_plot = st.session_state.current_context 

    if vector_to_plot is not None:
        labels = ['Energy', 'Valence', 'Dance', 'Tempo', 'Loud', 'Speech', 'Acoust', 'Instr', 'Live']
        vec = np.clip(vector_to_plot[:9], 0, 1)
        df_r = pd.DataFrame(dict(r=vec, theta=labels))
        fig = px.line_polar(df_r, r='r', theta='theta', line_close=True, range_r=[0, 1])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', polar=dict(bgcolor='rgba(0,0,0,0)', radialaxis=dict(visible=False), angularaxis=dict(color='#888')), showlegend=False, height=250, margin=dict(l=40, r=40, t=20, b=20))
        fig.update_traces(line_color='#1DB954', fill='toself', fillcolor='rgba(29, 185, 84, 0.15)', mode='lines+markers', marker=dict(size=6))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})