import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import sys
import time
import joblib
import json
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
ORACLE_META_PATH = os.path.join(DATA_DIR, 'oracle_meta.json')

# Carica variabili ambiente
load_dotenv()

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(
    page_title="Billie AI-lish", 
    layout="centered", 
    page_icon="🎵",
    initial_sidebar_state="expanded"
)

# --- HELPER PER GENERARE L'HTML DELLO SPLASH SCREEN ---
def get_splash_html(fade_out=False):
    # URL DELLA TUA FOTO (Collage Vinili)
    HERO_IMAGE_URL = "https://i.pinimg.com/1200x/e5/d0/bb/e5d0bb01bf836c60f236040cba62cb14.jpg"
    
    # Se fade_out è True, impostiamo opacità a 0 con transizione
    container_style = ""
    if fade_out:
        # Questa classe sovrascriverà lo stile base per creare la dissolvenza
        container_style = """
            opacity: 0 !important;
            pointer-events: none;
            transition: opacity 1.5s ease-out;
        """

    return f"""
    <style>
        /* Font Spotify-like (Circular) - Solo per Splash */
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700;900&display=swap');

        /* Copre interamente lo schermo */
        .splash-container {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            z-index: 99999;
            background-color: #000;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            overflow: hidden;
            opacity: 1; /* Stato iniziale visibile */
            {container_style}
        }}

        /* Sfondo Immagine Unica che "Respira" */
        .splash-background {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            
            background-image: url('{HERO_IMAGE_URL}');
            background-position: center center;
            background-repeat: no-repeat;
            background-size: cover;
            
            opacity: 0.4; 
            animation: kenburns 20s ease-in-out infinite alternate;
        }}

        @keyframes kenburns {{
            0% {{ transform: scale(1); }}
            100% {{ transform: scale(1.15); }}
        }}

        /* Testo BILLIE AI-LISH pulsante */
        .splash-title {{
            position: relative;
            z-index: 2;
            font-family: 'Montserrat', sans-serif;
            font-size: 5rem;
            font-weight: 900;
            color: #fff;
            text-transform: uppercase;
            letter-spacing: -2px;
            text-shadow: 0 4px 30px rgba(0,0,0,0.5);
            animation: fadein 1.5s ease-out;
        }}

        .splash-subtitle {{
            position: relative;
            z-index: 2;
            font-family: 'Montserrat', sans-serif;
            font-size: 1.1rem;
            font-weight: 700;
            color: #1DB954;
            letter-spacing: 2px;
            margin-top: 15px;
            text-transform: uppercase;
            animation: blink 2s infinite;
        }}

        @keyframes fadein {{
            0% {{ opacity: 0; transform: translateY(20px); }}
            100% {{ opacity: 1; transform: translateY(0); }}
        }}
        
        @keyframes blink {{
            0% {{ opacity: 0.6; }}
            50% {{ opacity: 1; }}
            100% {{ opacity: 0.6; }}
        }}
    </style>

    <div class="splash-container">
        <div class="splash-background"></div>
        <div class="splash-title">BILLIE AI-LISH</div>
        <div class="splash-subtitle">SCANSIONE DATASET IN CORSO...</div>
    </div>
    """


def render_splash_screen():
    # Se è la prima volta che entriamo nella sessione
    if 'first_load_done' not in st.session_state:
        placeholder = st.empty()
        # Renderizza stato iniziale (Visibile)
        placeholder.markdown(get_splash_html(fade_out=False), unsafe_allow_html=True)
        return placeholder
    return None


#cache recommender
@st.cache_resource(show_spinner=False)
def get_recommender():
    #Istanzia il recommender una sola volta (Streamlit rerun-safe)
    return SongRecommender()

# carica scaler
SCALER_FEATURES = None
try:
    payload = joblib.load(SCALER_PATH)
    if isinstance(payload, dict) and "scaler" in payload:
        scaler = payload.get("scaler")
        SCALER_FEATURES = payload.get("features")
    else:
        scaler = payload
except Exception:
    scaler = None

# CSS 
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

    /* 2. Tipografia Main (Font Standard) */
    .main-title { 
        font-size: 3.5rem; 
        font-weight: 900; 
        color: #FFFFFF; 
        text-transform: uppercase; 
        margin-bottom: 0; 
        letter-spacing: -1px;
    }
    .subtitle { 
        font-size: 1rem; 
        letter-spacing: 3px; 
        color: #1DB954; 
        text-transform: uppercase; 
        margin-bottom: 3rem; 
        font-weight: 700;
    }
    
    .track-name { 
        font-size: 3rem; 
        font-weight: 800; 
        margin-top: 1.5rem; 
        line-height: 1.1; 
        color: #fff; 
        letter-spacing: -1px;
    }
    .artist-name { 
        font-size: 1.6rem; 
        font-weight: 500; 
        color: #1DB954; 
        margin-bottom: 0.5rem; 
    }
    .meta-tag { font-size: 0.9rem; color: #888; letter-spacing: 1px; margin-bottom: 2rem; text-transform: uppercase; font-weight: 600;}
    .debug-tag { font-size: 0.7rem; color: #666; font-family: monospace; margin-top: -10px; margin-bottom: 20px;}

    /* 3. Spotify Player */
    .spotify-container {
        width: 100%;
        min-height: 352px;
        display: flex;
        justify-content: center;
        margin: 0 auto;
    }
    iframe {
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    }
    
    /* 4. Lista Feature a Destra (Track DNA) */
    .feature-list {
        text-align: left;
        background-color: #121212;
        padding: 20px;
        margin-left:-200px;
        border-radius: 8px;
        border: 1px solid #282828;
        height: 352px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .feature-item {
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
        border-bottom: 1px solid #282828;
        padding-bottom: 4px;
        font-size: 0.85rem;
    }
    .feat-label { color: #b3b3b3; font-weight: 600; }
    .feat-val { color: #1DB954; font-weight: 700;}

    /* 5. HISTORY TABLE */
    .history-container {
        width: 100% !important;
        max-width: 850px; 
        height: 250px !important; 
        overflow-y: auto; 
        margin: 20px auto; 
        padding: 0 15px;
        border-top: 1px solid #282828;
        border-bottom: 1px solid #282828;
        display: block;
    }
    
    .history-container::-webkit-scrollbar { width: 8px; }
    .history-container::-webkit-scrollbar-thumb { background: #555; border-radius: 4px; }
    .history-container::-webkit-scrollbar-thumb:hover { background: #888; }

    .history-table { 
        width: 100%; 
        border-collapse: collapse; 
        margin: 0 auto;
    }

    .history-table td {
        padding: 12px 10px;
        border-bottom: 1px solid #282828;
        font-size: 0.9rem;
        color: #b3b3b3;
        vertical-align: middle;
        text-align: left; 
    }

    .track-number { width: 40px; color: #1DB954; font-weight: bold; }
    .track-title-cell { color: #fff; font-weight: 500; letter-spacing: 0.5px; }
    .history-row-artist { color: #b3b3b3; font-weight: 400; font-size: 0.85rem; margin-left: 5px;}
    .history-table tr:hover td { background-color: #282828; }

    /* 6. Bottoni Main */
    section[data-testid="stMain"] .stButton > button {
        width: 100% !important;
        border-radius: 500px; /* Spotify pill shape */
        border: none;
        background-color: transparent;
        color: #fff;
        font-weight: 700;
        padding: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        border: 1px solid #555;
        transition: all 0.2s ease-in-out;
    }
    section[data-testid="stMain"] .stButton > button:hover { 
        border-color: #fff;
        transform: scale(1.04);
        background-color: rgba(255,255,255,0.1);
    }
    /* Pulsante Primario (Avvia Sessione / Genera) */
    div[data-testid="stVerticalBlock"] > div:nth-child(5) .stButton > button {
         background-color: #1DB954 !important;
         color: #000 !important;
         border: none !important;
    }
    div[data-testid="stVerticalBlock"] > div:nth-child(5) .stButton > button:hover {
         background-color: #1ed760 !important;
         transform: scale(1.02);
    }

    </style>
""", unsafe_allow_html=True)

# INIZIALIZZAZIONE

# mostra splash screen (schermata iniziale con foto canzoni)
loading_placeholder = render_splash_screen()

# esegue i caricamenti durante la schermata
if 'oracle' not in st.session_state:
    
    # Simula caricamento minimo se è troppo veloce
    # time.sleep(1.0) 

    try:
        if os.path.exists(ORACLE_PATH):
            st.session_state.oracle = joblib.load(ORACLE_PATH)
        else:
            st.session_state.oracle = MusicOracle()
    except Exception:
        st.session_state.oracle = MusicOracle()

    try:
        # Questo è il passaggio più lento (carica CSV e crea Matrice)
        st.session_state.recommender = get_recommender()
    except Exception as e:
        if loading_placeholder: loading_placeholder.empty()
        st.error(f"System Error: {e}")
        st.stop()
        
    # Inizializzazione variabili sessione
    st.session_state.past_track_ids = []
    st.session_state.suggestion_made = False
    st.session_state.current_track = None
    st.session_state.predicted_vector = None
    st.session_state.recs_queue = pd.DataFrame()
    st.session_state.vibe_history = [50]
    st.session_state.session_blacklist = []

# effetto dissolvenza
if loading_placeholder is not None:
    
    time.sleep(1.5)
    
    
    loading_placeholder.markdown(get_splash_html(fade_out=True), unsafe_allow_html=True)
    
    
    time.sleep(1.5)
    
    # Rimuoviamo il componente
    loading_placeholder.empty()
    st.session_state.first_load_done = True

# funzione generazione
def generate_new_recommendation(manual_target=None):
    if st.session_state.history_df is not None:
        try:
            # Se c'è un target manuale o la coda è vuota, ricalcoliamo tutto
            if manual_target is not None or st.session_state.recs_queue.empty:
                st.session_state.recs_queue = pd.DataFrame() # Reset coda
                recs_df, pred_vector = st.session_state.recommender.recommend(
                    st.session_state.history_df, 
                    k=30, 
                    target_features=manual_target,
                    # Passiamo la blacklist della sessione al Recommender
                    session_blacklist=st.session_state.session_blacklist 
                )
            # se vi sono canzoni nella coda, usiamo quelle 
            else:
                recs_df = st.session_state.recs_queue
                # Manteniamo il vettore target precedente
                pred_vector = st.session_state.predicted_vector if st.session_state.predicted_vector is not None else np.zeros(9)

            # controllo per evitare crash
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

# funzione salvataggio feedback (like dislike)
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
        # Se il file non esiste, scrivi header. Se esiste, appendi senza header.
        df_new.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
    except Exception as e:
        st.warning(f"Impossibile salvare feedback su {os.path.basename(csv_path)}: {e}")

# AGGIORNAMENTO VIBE
def update_vibe(points):
    """Aggiorna il grafico della vibe (0-100)"""
    current_score = st.session_state.vibe_history[-1]
    new_score = np.clip(current_score + points, 0, 100)
    st.session_state.vibe_history.append(new_score)
    # Teniamo solo gli ultimi 50 punti per il grafico
    if len(st.session_state.vibe_history) > 50:
        st.session_state.vibe_history.pop(0)

#CARICO LA CRONOLOGIA UTENTE    

@st.cache_data(show_spinner=False)
def load_data(history_path: str) -> pd.DataFrame:
    """Carica lo storico da CSV (se esiste)."""
    if os.path.exists(history_path):
        return pd.read_csv(history_path)
    return pd.DataFrame()
# FUNZIONE DEBUG
def recalculate_user_stats():
    """
    Ricalcola Top Artist/Genre sulle ultime 50 righe e stampa un REPORT DI DEBUG
    per verificare se stiamo guardando le canzoni giuste.
    """
    if 'history_df' in st.session_state and st.session_state.history_df is not None:
        df = st.session_state.history_df
        
        if not df.empty:
            #  DEBUG: Identifichiamo il blocco analizzato
            recent = df.tail(50)
            
            # Stampa nel terminale cosa sta guardando l'AI
            print("\n" + "="*40)
            print(f" ANALISI 'MOOD ATTUALE' (Ultime {len(recent)} righe)")
            print(f" Prima canzone del blocco: {recent.iloc[0]['name']} - {recent.iloc[0]['artist']}")
            print(f" Ultima canzone del blocco: {recent.iloc[-1]['name']} - {recent.iloc[-1]['artist']}")
            print("="*40)

            # CALCOLO TOP ARTIST
            try:
                # Normalizza e pulisci
                artists = recent['artist'].astype(str).dropna()
                artists = artists[~artists.str.lower().isin(['unknown', 'nan', '', '[]'])]
                # Rimuovi parentesi e virgolette
                artists = artists.apply(lambda x: str(x).replace("['", "").replace("']", "").replace("'", "").replace('"', "").strip())
                
                if not artists.empty:
                    counts = artists.value_counts()
                    st.session_state.top_artist = counts.index[0]
                    print(f" Artista Vincente: {st.session_state.top_artist} ({counts.iloc[0]} ascolti)")
                else:
                    st.session_state.top_artist = "N/A"
            except Exception as e:
                print(f" Errore artista: {e}")
                st.session_state.top_artist = "-"

            # CALCOLO GENERE TOP
            try:
                genres = recent['genres'].astype(str).dropna()
                genres = genres[~genres.str.lower().isin(['[]', 'unknown', 'nan', ''])]
                
                if not genres.empty:
                    clean_genres = genres.apply(lambda x: str(x).replace("['", "").replace("']", "").replace("'", "").title())
                    g_counts = clean_genres.value_counts()
                    st.session_state.top_genre = g_counts.index[0]
                    print(f" Genere Vincente: {st.session_state.top_genre}")
                else:
                    st.session_state.top_genre = "N/A"
            except:
                st.session_state.top_genre = "-"
            print("="*40 + "\n")
            
    else:
        st.session_state.top_artist = "-"
        st.session_state.top_genre = "-"

# ORACLE PER TRAINING INCREMENTALE
def _load_oracle_meta() -> dict:
    try:
        if os.path.exists(ORACLE_META_PATH):
            with open(ORACLE_META_PATH, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_oracle_meta(meta: dict) -> None:
    try:
        with open(ORACLE_META_PATH, 'w') as f:
            json.dump(meta, f)
    except Exception:
        pass


# FUNZIONE RIADDESTRAMENTO ORACLE SU NUOVE CANZONI 
def retrain_oracle_on_new_songs():
    """
    Riaddestra l'oracle solo sulle nuove canzoni che non ha mai visto.
    Usa il flag oracle_trained_up_to_song per tracciare l'ultima canzone vista.
    """
    # Se non c'è cronologia o oracle, esci
    if st.session_state.history_df is None or st.session_state.history_df.empty:
        # Meta consistency: if empty, reset meta file
        _save_oracle_meta({})
        return
    if not st.session_state.get('oracle'):
        return

    # Prepara i dati
    df = st.session_state.history_df.copy()

    # Ordina per tempo (se disponibile)
    if 'played_at' in df.columns:
        df['played_at'] = pd.to_datetime(df['played_at'], errors='coerce', utc=True, format='mixed')
        df = df.dropna(subset=['played_at'])
        df = df.sort_values(by='played_at', ascending=True).reset_index(drop=True)

    # Colonne audio
    feature_cols = ['energy', 'valence', 'danceability', 'tempo',
                    'loudness', 'speechiness', 'acousticness', 'instrumentalness',
                    'liveness']

    # Verifica che le colonne esistano
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"Colonne mancanti per training: {missing}")
        return

    meta = _load_oracle_meta()

    # Preferiamo il meta su disco (persistente). Fallback: inferiamo dai passi di training.
    total_songs = len(df)
    last_rowcount = meta.get('last_trained_rowcount')
    if isinstance(last_rowcount, int) and last_rowcount > 0:
        last_trained_idx = int(last_rowcount) - 1
    else:
        trained_steps = len(getattr(st.session_state.oracle, 'loss_history', []) or [])
        last_trained_idx = trained_steps if trained_steps > 0 else -1

    # Clamp per sicurezza
    last_trained_idx = max(-1, min(last_trained_idx, total_songs - 1))
    st.session_state.oracle_trained_up_to_song = last_trained_idx

    new_songs_count = total_songs - (last_trained_idx + 1)
    if new_songs_count <= 0:
        print("Nessuna nuova canzone da addestrare")
        return

    print(f"\n{'='*60}")
    print(f"RIADDESTRAMENTO ORACLE SU {new_songs_count} NUOVE CANZONI")
    print(f"{'='*60}")

    # Recupera contesto dall'ultima sessione, se disponibile
    current_context = None
    last_ctx = meta.get('last_context')
    if isinstance(last_ctx, list) and len(last_ctx) == len(feature_cols):
        try:
            current_context = np.array(last_ctx, dtype=float)
        except Exception:
            current_context = None

    if last_trained_idx == -1:
        # Primo addestramento: contesto iniziale = prima canzone
        current_context = df.loc[0, feature_cols].values.astype(float)
        start_idx = 1
    else:
        start_idx = last_trained_idx + 1
        # Se non abbiamo il contesto salvato, ricostruiamolo fino a last_trained_idx
        if current_context is None:
            current_context = df.loc[0, feature_cols].values.astype(float)
            for j in range(1, last_trained_idx + 1):
                tgt_j = df.loc[j, feature_cols].values.astype(float)
                current_context = calculate_avalanche_context(current_context, tgt_j, n=5)

    # Loop di addestramento SOLO sulle nuove canzoni
    for i in range(start_idx, total_songs):
        target_track = df.loc[i, feature_cols].values
        st.session_state.oracle.train_incremental(current_context, target_track)
        current_context = calculate_avalanche_context(current_context, target_track, n=5)

    # Aggiorna il flag: ora l'oracle ha visto fino all'ultima canzone
    st.session_state.oracle_trained_up_to_song = total_songs - 1

    # Salva oracle su disco
    try:
        joblib.dump(st.session_state.oracle, ORACLE_PATH)

        # Salva anche meta: quante righe viste + ultimo contesto (per evitare retrain completo al prossimo avvio)
        _save_oracle_meta({
            'last_trained_rowcount': int(total_songs),
            'last_context': [float(x) for x in np.array(current_context, dtype=float).tolist()],
        })

        print(f"Oracle salvato. Totale interazioni: {len(st.session_state.oracle.loss_history)}")
    except Exception as e:
        print(f"Errore nel salvataggio dell'oracle: {e}")

    print(f"{'='*60}\n")


# NOTE: Rimossa sezione AUTO-FETCH per migliorare UX

if 'history_df' not in st.session_state:
    try:
        history_df = load_data(HISTORY_PATH)
        
        # Se il file esiste ed è stato caricato
        if history_df is not None and not history_df.empty:
            
            # NORMALIZZAZIONE COLONNE
            
            history_df.columns = history_df.columns.astype(str).str.lower().str.strip()
            
            # REMAP COLONNE PER EVITARE ERRORI
            rename_map = {
                'artists': 'artist', 
                'artist_name': 'artist',
                'genre': 'genres', 
                'track_name': 'name', 
                'song': 'name', 
                'track': 'name'
            }
            history_df.rename(columns=rename_map, inplace=True)
            
            #Verifica esistenza 'artist' (Evita il KeyError)
            if 'artist' not in history_df.columns:
                # Se manca ancora, crea colonna dummy per non crashare
                history_df['artist'] = "Unknown"
            
            # 4. Verifica esistenza 'genres'
            if 'genres' not in history_df.columns:
                history_df['genres'] = "[]"

            # lo assegna allo stato (dopo che sono stati evitati eventuali errori)
            st.session_state.history_df = history_df
            
            # calcolo contesto e statistiche
            features = st.session_state.recommender.audio_cols
            valid = [c for c in features if c in history_df.columns]
            
            if valid:
                st.session_state.current_context = history_df[valid].mean().values
                st.session_state.song_count = len(history_df)
                
                # calcolo il genere top
                v_gen = history_df[~history_df['genres'].isin(['unknown', 'nan', '[]'])]['genres']
                st.session_state.top_genre = v_gen.mode()[0].title() if not v_gen.empty else "N/A"
                
                # calcolo artista top
                v_art = history_df[~history_df['artist'].isin(['unknown', 'nan', 'Unknown'])]['artist']
                st.session_state.top_artist = v_art.mode()[0] if not v_art.empty else "N/A"
            else:
                # Dati anagrafici ok, ma audio mancante
                st.session_state.current_context = np.array([0.5] * 9)
                st.session_state.song_count = len(history_df)
                st.session_state.top_artist = "N/A"
                st.session_state.top_genre = "N/A"

        else:
            # File vuoto o inesistente: inizializza vuoto
            st.session_state.history_df = pd.DataFrame()
            st.session_state.current_context = np.array([0.5] * 9)
            st.session_state.song_count = 0
            st.session_state.top_artist = "-"
            st.session_state.top_genre = "-"
        recalculate_user_stats()
    except Exception:
        # Fallback totale per evitare crash
        st.session_state.history_df = None
        st.session_state.current_context = np.array([0.5] * 9)
        st.session_state.song_count = 0
        st.session_state.top_artist = "-"
        st.session_state.top_genre = "-"

# header
st.markdown("<div class='main-title'>BILLIE AI-LISH</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Artificial Music Agent</div>", unsafe_allow_html=True)


# sidebar
st.sidebar.header("CONTROL ROOM")

# Sezione Oracle
if st.session_state.get('oracle') and hasattr(st.session_state.oracle, 'loss_history'):
    history = st.session_state.oracle.loss_history
    n_trained = len(history)
    
    st.sidebar.success(f"Oracle: {n_trained} interazioni")
    
    if n_trained > 1:
        first_loss = history[0]
        last_loss = history[-1]
        if first_loss != 0:
            improvement = ((first_loss - last_loss) / first_loss) * 100
        else:
            improvement = 0.0
            
        st.sidebar.caption(f"Loss iniziale: {first_loss:.4f}")
        st.sidebar.caption(f"Loss attuale: {last_loss:.4f}")
        
        if improvement > 0:
            st.sidebar.caption(f"Miglioramento: {improvement:.1f}%")
        else:
            st.sidebar.caption(f"Loss in crescita (Adattamento)")

st.sidebar.markdown("---")

# Sezione Mood Utente
# Usiamo le variabili di stato aggiornate dalla funzione sopra
t_genre = st.session_state.get('top_genre', 'Calcolo...')
t_artist = st.session_state.get('top_artist', 'Calcolo...')

st.sidebar.markdown(f"**Mood Attuale (Last 50):**")
st.sidebar.caption(f"🎵 Genere: {t_genre}")
st.sidebar.caption(f"🎤 Artista: {t_artist}")

# indicatore coda (brani che vengono proposti all'utente)
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
            
            # Ricarica i dati
            history_df = load_data(HISTORY_PATH)
            if history_df is not None and not history_df.empty:
                # Normalizza
                history_df.columns = history_df.columns.astype(str).str.lower().str.strip()
                rename_map = {'artists': 'artist', 'artist_name': 'artist', 'genre': 'genres', 'track_name': 'name', 'song': 'name', 'track': 'name'}
                history_df.rename(columns=rename_map, inplace=True)
                if 'artist' not in history_df.columns: history_df['artist'] = "Unknown"
                if 'genres' not in history_df.columns: history_df['genres'] = "[]"
                st.session_state.history_df = history_df
                
                #Riaddestra oracle sulle nuove canzoni
                #retrain_oracle_on_new_songs()
            
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Errore: {e}")

# generazione
c1, col_gen, c3 = st.columns([1, 2, 1])
with col_gen:
    
    # calcola lo stato della coda
    if 'recs_queue' in st.session_state:
        q_len = len(st.session_state.recs_queue)
    else:
        q_len = 0
        
    is_start_session = (q_len == 0)
    btn_label = "AVVIA SESSIONE" if is_start_session else "GENERA PROSSIMA"
    
    #  CREA UNO SPAZIO VUOTO PER IL BOTTONE (placeholder)
    # Questo ci permette di cancellarlo dopo il click
    btn_placeholder = st.empty()
    
    # 3. DISEGNA IL BOTTONE DENTRO IL PLACEHOLDER
    clicked = btn_placeholder.button(btn_label, type="primary", key="final_gen_key")
    
    if clicked:
        btn_placeholder.empty() 
        
        
        if is_start_session:
            with st.spinner("Sintonizzazione AI in corso..."):
                try:
                    fetch_history()
                    load_data.clear() 
                    
                    history_df = load_data(HISTORY_PATH)
                    
                    if history_df is not None and not history_df.empty:
                        history_df.columns = history_df.columns.astype(str).str.lower().str.strip()
                        rename_map = {'artists': 'artist', 'artist_name': 'artist', 'genre': 'genres', 'track_name': 'name', 'song': 'name', 'track': 'name'}
                        history_df.rename(columns=rename_map, inplace=True)
                        if 'artist' not in history_df.columns: history_df['artist'] = "Unknown"
                        if 'genres' not in history_df.columns: history_df['genres'] = "[]"
                        
                        st.session_state.history_df = history_df
                        
                        features = st.session_state.recommender.audio_cols
                        valid = [c for c in features if c in history_df.columns]
                        if valid:
                            st.session_state.current_context = history_df[valid].mean().values
                            st.session_state.song_count = len(history_df)

                        recalculate_user_stats()
                        
                        #Riaddestra oracle sulle nuove canzoni
                        retrain_oracle_on_new_songs()
                
                except Exception as e:
                    st.error(f"Errore critico: {e}")
                    st.stop()
                
                if generate_new_recommendation():
                    time.sleep(0.5) 
                    st.rerun()
                else:
                    st.error("Nessun risultato trovato. Riprova.")

        #  PROSSIMA CANZONE
        else:
            # Anche qui il bottone è sparito, quindi non puoi cliccarlo due volte
            with st.spinner("Generazione..."):
                generate_new_recommendation()
                st.rerun()

#DISPLAY CANZONE 
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

        # INIZIALIZZAZIONE
        real_data_map = {} 
        features_for_scaler = SCALER_FEATURES if SCALER_FEATURES else audio_cols
        
        # Tentativo di recupero valori reali dallo scaler (se disponibile)
        try:
            if scaler:
                # Creiamo il vettore normalizzato
                norm_vec = np.array([track.get(c, 0) for c in features_for_scaler]).reshape(1, -1)
                # Tentiamo l'inversione
                real_vec = scaler.inverse_transform(norm_vec)[0]
                real_data_map = dict(zip(features_for_scaler, real_vec))
        except:
            pass # Se fallisce, useremo le formule di fallback nel loop

        # GENERAZIONE LISTA VISUALE 
        display_feats = audio_cols 
        html_stats = "<div class='feature-list'><div style='color:#fff; font-weight:900; margin-bottom:10px; text-transform:uppercase; letter-spacing:2px; font-size:0.9rem;'>Track DNA</div>"
        
        for f in display_feats:
            val = track.get(f, 0)
            
            # LOGICA VISUALIZZAZIONE 
            if f == 'tempo':
                # Valore già reale (es. 123.8) -> Usa quello
                if val > 2.0:
                    val_s = f"{int(val)} BPM"
                # Valore normalizzato -> Prova Scaler -> Fallback Formula
                else:
                    bpm = real_data_map.get(f, val * 160 + 40)
                    val_s = f"{int(bpm)} BPM"
            
            elif f == 'loudness':
                # Valore già in dB (es. -8.5) -> Usa quello
                if val < -1.0 or val > 1.0: 
                     val_s = f"{val:.1f} dB"
                #  Valore normalizzato -> Prova Scaler -> Fallback Formula
                else:
                     db = real_data_map.get(f, val * 60 - 60)
                     val_s = f"{db:.1f} dB"
            
            else:
                # Percentuali (Energy, Valence, ecc.)
                # Gestisce sia 0-1 che 0-100
                if val > 1.0:
                    val_s = f"{int(val)}"
                else:
                    val_s = f"{int(val * 100)}"

            html_stats += f"<div class='feature-item'><span class='feat-label'>{f.capitalize()}</span><span class='feat-val'>{val_s}</span></div>"
        html_stats += "</div>"
        st.markdown(html_stats, unsafe_allow_html=True)

    st.markdown(f"<div class='track-name'>{track['name']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='artist-name'>{track['artist']}</div>", unsafe_allow_html=True)
    g_str = str(track.get('genres', '')).replace("['", "").replace("']", "").replace("'", "").title()
    if g_str == "[]" or g_str == "Unknown": g_str = "Genere non classificato"
    
    y = str(int(float(track.get('year', 0)))) if track.get('year') else ""
    st.markdown(f"<div class='meta-tag'>{y} • {g_str}</div>", unsafe_allow_html=True)
    
    # DEBUG TAG VISIBILE A SCHERMO
    reason = track.get('reason_text', 'Algoritmo')
    match_score = track.get('match_percentage', 0)
    st.markdown(f"<div class='debug-tag'>[DEBUG AI] Motivo: {reason} | Match Audio: {match_score}%</div>", unsafe_allow_html=True)

    #BOTTONI (Like, Dislike, Save)
    c_dislike, c_like, c_save = st.columns([1, 1, 2])
    
    with c_dislike:
        if st.button("DISLIKE", key="btn_dislike"):
            update_vibe(-10) # Scende Vibe

            real_g, real_p = get_track_details(track['id'])
            append_feedback_csv(DISLIKED_PATH, track, real_g, real_p)

            st.session_state.session_blacklist.append(track['id'])
            st.session_state.past_track_ids.append(str(track['id']))
            
            # NOTA: Non azzeriamo la coda qui, proseguiamo con la prossima
            generate_new_recommendation()
            st.rerun()

    # LIKE (Sale Vibe, Allena AI, Blacklist Sessione, Next)
    with c_like:
        if st.button("LIKE", key="btn_like"):
            update_vibe(+10) # Sale Vibe

            real_g, real_p = get_track_details(track['id'])
            append_feedback_csv(LIKED_PATH, track, real_g, real_p)

            #  Allenamento Oracle
            cols = st.session_state.recommender.audio_cols
            feats = np.array([track[k] for k in cols])
            if st.session_state.oracle:
                print("\n" + "="*60)
                print(f"LIKE - {track['name']} by {track['artist']}")
                print("="*60)
                st.session_state.oracle.train_incremental(st.session_state.current_context, feats)
                try:
                    joblib.dump(st.session_state.oracle, ORACLE_PATH)
                except Exception: pass

            # Aggiorna Contesto
            st.session_state.current_context = calculate_avalanche_context(st.session_state.current_context, feats, st.session_state.song_count)

            st.session_state.session_blacklist.append(track['id'])
            st.session_state.past_track_ids.append(str(track['id']))
            
            # NOTA: Non azzeriamo la coda qui, proseguiamo con la prossima
            generate_new_recommendation() 
            st.rerun()

    #  SAVE (Sale Vibe Molto, Allena AI, Salva File, Next)
    with c_save:
        if st.button("SALVA IN PLAYLIST", key="btn_save"):
            with st.status("Salvataggio...", expanded=False) as status:
                update_vibe(+20) # Vibe sale molto
                
                real_g, real_p = get_track_details(track['id'])
                cols = st.session_state.recommender.audio_cols
                feats = np.array([track[k] for k in cols])
                
                if st.session_state.oracle:
                    print("\n" + "="*60)
                    print(f"SAVE - {track['name']} by {track['artist']}")
                    print("="*60)   
                    st.session_state.oracle.train_incremental(st.session_state.current_context, feats)
                    try:
                        joblib.dump(st.session_state.oracle, ORACLE_PATH)
                    except Exception: pass
                
                st.session_state.song_count += 1
                st.session_state.current_context = calculate_avalanche_context(st.session_state.current_context, feats, st.session_state.song_count)
                
                if pd.notna(tid):
                    add_track_to_playlist(str(tid))
                
                st.session_state.session_blacklist.append(track['id'])
                
                #  Salva SOLO su CSV Playlist (DISCO)
                new_row = {'id': track['id'], 'name': track['name'], 'artist': track['artist'], 'genres': real_g, 'popularity': real_p, 'year': track.get('year'), **{k: track[k] for k in cols}}
                df_new = pd.DataFrame([new_row])
                df_new.to_csv(PLAYLIST_SAVED_PATH, mode='a', header=not os.path.exists(PLAYLIST_SAVED_PATH), index=False)
                
                #  AGGIORNA MEMORIA SESSIONE (RAM) - NON SU DISCO
                st.session_state.history_df = pd.concat([st.session_state.history_df, df_new], ignore_index=True)

                # NOTA: Non azzeriamo la coda qui, proseguiamo con la prossima
                generate_new_recommendation()
                status.update(label="Salvato!", state="complete")
            st.rerun()

st.markdown("---")

# GRAFICO VIBE DELLA SESSIONE
st.markdown("<div style='letter-spacing: 2px; font-weight: 900; color: #444; margin-top:20px; font-size: 0.8rem;'>SESSION VIBE</div>", unsafe_allow_html=True)
vibe_data = pd.DataFrame(st.session_state.vibe_history, columns=['Vibe'])
st.line_chart(vibe_data, height=150, color='#1DB954')

#  HISTORY E RADAR FEATURES
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
    
    # Recupera il vettore target
    vector_source = st.session_state.predicted_vector
    if vector_source is None and st.session_state.current_context is not None:
        vector_source = st.session_state.current_context 

    if vector_source is not None:
        # Creiamo una copia per normalizzarla SOLO per il grafico (senza toccare i dati veri)
        # Assicuriamoci che sia float per evitare errori di divisione
        plot_vec = np.array(vector_source[:9], dtype=float).copy()
        
        # NORMALIZZAZIONE PER IL GRAFICO
        # Indici: 0=Energy, 1=Valence, 2=Dance, 3=Tempo, 4=Loud, 5=Speech, 6=Acoust, 7=Instr, 8=Live
        
        #  TEMPO (Index 3): Se è "reale" (es. 120), portalo a 0-1
        # Assumiamo un range tipico 40-200 BPM
        if plot_vec[3] > 2.0: 
            plot_vec[3] = (plot_vec[3] - 40) / 160.0
        
        #  LOUDNESS (Index 4): Se è in dB (es. -10), portalo a 0-1
        # Assumiamo un range tipico -60dB a 0dB
        if plot_vec[4] < 0:
            plot_vec[4] = (plot_vec[4] + 60) / 60.0
            
        #  ALTRE FEATURE (0,1,2,5,6,7,8): Se sono in scala 0-100, dividi per 100
        # Se sono già 0-1, restano uguali.
        for i in [0, 1, 2, 5, 6, 7, 8]:
            if plot_vec[i] > 1.0:
                plot_vec[i] = plot_vec[i] / 100.0

        # Clipping finale di sicurezza per restare nel grafico (0.0 - 1.0)
        vec = np.clip(plot_vec, 0, 1)
        
        # CREAZIONE GRAFICO 
        labels = ['Energy', 'Valence', 'Dance', 'Tempo', 'Loud', 'Speech', 'Acoust', 'Instr', 'Live']
        df_r = pd.DataFrame(dict(r=vec, theta=labels))
        
        fig = px.line_polar(df_r, r='r', theta='theta', line_close=True, range_r=[0, 1])
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            polar=dict(
                bgcolor='rgba(0,0,0,0)', 
                radialaxis=dict(visible=False), 
                angularaxis=dict(color='#888')
            ), 
            showlegend=False, 
            height=250, 
            margin=dict(l=40, r=40, t=20, b=20)
        )
        fig.update_traces(line_color='#1DB954', fill='toself', fillcolor='rgba(29, 185, 84, 0.15)', mode='lines+markers', marker=dict(size=6))
        
        st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})