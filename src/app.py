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
        <div class="splash-subtitle">INITIALIZING NEURAL AUDIO ENGINE...</div>
    </div>
    """

# --- FUNZIONE PRINCIPALE SPLASH ---
def render_splash_screen():
    # Se è la prima volta che entriamo nella sessione
    if 'first_load_done' not in st.session_state:
        placeholder = st.empty()
        # Renderizza stato iniziale (Visibile)
        placeholder.markdown(get_splash_html(fade_out=False), unsafe_allow_html=True)
        return placeholder
    return None

# --- CACHED RECOMMENDER FACTORY ---
@st.cache_resource(show_spinner=False)
def get_recommender():
    """Istanzia il recommender una sola volta (Streamlit rerun-safe)."""
    return SongRecommender()

# --- CARICAMENTO SCALER ---
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

# --- CSS GENERALE (FONT STANDARD STREAMLIT) ---
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

# --- 1. INIZIALIZZAZIONE ---

# A. MOSTRA SPLASH SCREEN (Se necessario)
loading_placeholder = render_splash_screen()

# B. ESEGUI I CARICAMENTI PESANTI (Sotto lo Splash)
if 'oracle' not in st.session_state:
    
    # Simula caricamento minimo se è troppo veloce, per godersi l'intro
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

# C. DISSOLVENZA E CHIUSURA SPLASH SCREEN
if loading_placeholder is not None:
    # 1. Attendiamo un attimo per assicurare che l'utente abbia visto il logo
    time.sleep(1.5)
    
    # 2. Aggiorniamo l'HTML con la versione che ha Opacity: 0 (Fade Out)
    loading_placeholder.markdown(get_splash_html(fade_out=True), unsafe_allow_html=True)
    
    # 3. Attendiamo la durata della transizione CSS (1.5s)
    time.sleep(1.5)
    
    # 4. Rimuoviamo il componente
    loading_placeholder.empty()
    st.session_state.first_load_done = True

# --- FUNZIONE GENERAZIONE (BUFFERIZZATA + FILTRO BLACKLIST) ---
def generate_new_recommendation(manual_target=None):
    if st.session_state.history_df is not None:
        try:
            # 1. Se c'è un target manuale (Slider) o la coda è vuota, ricalcoliamo tutto
            if manual_target is not None or st.session_state.recs_queue.empty:
                st.session_state.recs_queue = pd.DataFrame() # Reset coda
                recs_df, pred_vector = st.session_state.recommender.recommend(
                    st.session_state.history_df, 
                    k=30, 
                    target_features=manual_target,
                    # Passiamo la blacklist della sessione al Recommender
                    session_blacklist=st.session_state.session_blacklist 
                )
            # 2. Se la coda ha canzoni, usiamo quelle (ISTANTANEO)
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
        # Se il file non esiste, scrivi header. Se esiste, appendi senza header.
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
# --- FUNZIONE DI CALCOLO E DEBUG ---
def recalculate_user_stats():
    """
    Ricalcola Top Artist/Genre sulle ultime 50 righe e stampa un REPORT DI DEBUG
    per verificare se stiamo guardando le canzoni giuste.
    """
    if 'history_df' in st.session_state and st.session_state.history_df is not None:
        df = st.session_state.history_df
        
        if not df.empty:
            # 🕵️ DEBUG: Identifichiamo il blocco analizzato
            recent = df.tail(50)
            
            # Stampa nel terminale cosa sta guardando l'AI
            print("\n" + "="*40)
            print(f" ANALISI 'MOOD ATTUALE' (Ultime {len(recent)} righe)")
            print(f" Prima canzone del blocco: {recent.iloc[0]['name']} - {recent.iloc[0]['artist']}")
            print(f" Ultima canzone del blocco: {recent.iloc[-1]['name']} - {recent.iloc[-1]['artist']}")
            print("="*40)

            # --- 1. CALCOLO TOP ARTIST ---
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

            # --- 2. CALCOLO TOP GENRE ---
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


# NOTE: Rimossa sezione AUTO-FETCH per migliorare UX

if 'history_df' not in st.session_state:
    try:
        history_df = load_data(HISTORY_PATH)
        
        # Se il file esiste ed è stato caricato
        if history_df is not None and not history_df.empty:
            
            # --- FIX CRUCIALE: NORMALIZZAZIONE COLONNE ---
            # 1. Tutto minuscolo e senza spazi
            history_df.columns = history_df.columns.astype(str).str.lower().str.strip()
            
            # 2. Rinomina colonne problematiche (artists -> artist, genre -> genres)
            rename_map = {
                'artists': 'artist', 
                'artist_name': 'artist',
                'genre': 'genres', 
                'track_name': 'name', 
                'song': 'name', 
                'track': 'name'
            }
            history_df.rename(columns=rename_map, inplace=True)
            
            # 3. Verifica esistenza 'artist' (Evita il KeyError)
            if 'artist' not in history_df.columns:
                # Se manca ancora, crea colonna dummy per non crashare
                history_df['artist'] = "Unknown"
            
            # 4. Verifica esistenza 'genres'
            if 'genres' not in history_df.columns:
                history_df['genres'] = "[]"

            # Ora è sicuro assegnarlo allo stato
            st.session_state.history_df = history_df
            
            # --- Calcolo Contesto e Statistiche ---
            features = st.session_state.recommender.audio_cols
            valid = [c for c in features if c in history_df.columns]
            
            if valid:
                st.session_state.current_context = history_df[valid].mean().values
                st.session_state.song_count = len(history_df)
                
                # Calcolo Top Genre (sicuro perché la colonna esiste per forza ora)
                v_gen = history_df[~history_df['genres'].isin(['unknown', 'nan', '[]'])]['genres']
                st.session_state.top_genre = v_gen.mode()[0].title() if not v_gen.empty else "N/A"
                
                # Calcolo Top Artist (sicuro perché la colonna esiste per forza ora)
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

# --- HEADER ---
st.markdown("<div class='main-title'>BILLIE AI-LISH</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Artificial Music Agent</div>", unsafe_allow_html=True)


# --- SIDEBAR (VISUALIZZAZIONE) ---
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

# --- INDICATORE CODA ---
q_len = len(st.session_state.recs_queue) if 'recs_queue' in st.session_state else 0
if q_len > 0:
    st.sidebar.success(f"Coda Veloce Attiva: {q_len} brani pronti")
else:
    st.sidebar.info("Coda vuota (Il prossimo click calcolerà un nuovo batch)")

# --- TASTO RESET ---
if st.sidebar.button(" RESETTA CERVELLO AI", type="primary"):
    st.session_state.recs_queue = pd.DataFrame()
    st.session_state.session_blacklist = []
    # Rimuove cache dati
    st.cache_data.clear()
    st.rerun()

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
    
    # 1. Calcola lo stato: siamo all'inizio (0) o stiamo già ascoltando (>0)?
    q_len = len(st.session_state.recs_queue) if 'recs_queue' in st.session_state else 0
    is_start_session = (q_len == 0)
    
    # 2. Decidi l'etichetta del bottone PRIMA di crearlo
    btn_label = "AVVIA SESSIONE" if is_start_session else "GENERA PROSSIMA"
    
    # 3. CREA L'UNICO BOTTONE (Questo è l'unico st.button in questa colonna)
    pressed = st.button(btn_label, type="primary", key="main_gen_btn")
    
    if pressed:
        # --- LOGICA A: AVVIO SESSIONE (Caricamenti Pesanti) ---
        if is_start_session:
            # Usa st.status per raggruppare i messaggi e non spaccare il layout
            with st.status("Avvio Sessione...", expanded=True) as status:
                
                status.write(" Connessione a Spotify...")
                try:
                    # A. Scarica Dati
                    fetch_history()
                    load_data.clear() # Pulisce la cache
                    
                    # B. Ricarica DataFrame
                    history_df = load_data(HISTORY_PATH)
                    
                    if history_df is not None and not history_df.empty:
                        # Normalizzazione
                        history_df.columns = history_df.columns.astype(str).str.lower().str.strip()
                        rename_map = {'artists': 'artist', 'artist_name': 'artist', 'genre': 'genres', 'track_name': 'name', 'song': 'name', 'track': 'name'}
                        history_df.rename(columns=rename_map, inplace=True)
                        if 'artist' not in history_df.columns: history_df['artist'] = "Unknown"
                        if 'genres' not in history_df.columns: history_df['genres'] = "[]"
                        
                        st.session_state.history_df = history_df
                        
                        # C. Calcola Contesto
                        features = st.session_state.recommender.audio_cols
                        valid = [c for c in features if c in history_df.columns]
                        if valid:
                            st.session_state.current_context = history_df[valid].mean().values
                            st.session_state.song_count = len(history_df)

                        # D. Aggiorna Statistiche (Kid Yugi ecc.)
                        recalculate_user_stats()
                
                except Exception as e:
                    st.error(f"Errore critico: {e}")
                    status.update(label="Errore", state="error")
                    st.stop()
                
                # E. Genera Primo Batch
                status.write(" Analisi DNA Musicale...")
                if generate_new_recommendation():
                    status.update(label="Sessione Avviata!", state="complete")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    status.update(label="Nessun risultato trovato", state="error")

        # --- LOGICA B: PROSSIMA CANZONE (Istantaneo) ---
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

        # --- 1. INIZIALIZZAZIONE SICURA ---
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

        # --- 2. GENERAZIONE LISTA VISUALE ---
        display_feats = audio_cols 
        html_stats = "<div class='feature-list'><div style='color:#fff; font-weight:900; margin-bottom:10px; text-transform:uppercase; letter-spacing:2px; font-size:0.9rem;'>Track DNA</div>"
        
        for f in display_feats:
            val = track.get(f, 0)
            
            # --- LOGICA INTELLIGENTE DI VISUALIZZAZIONE ---
            if f == 'tempo':
                # Caso 1: Valore già reale (es. 123.8) -> Usa quello
                if val > 2.0:
                    val_s = f"{int(val)} BPM"
                # Caso 2: Valore normalizzato -> Prova Scaler -> Fallback Formula
                else:
                    bpm = real_data_map.get(f, val * 160 + 40)
                    val_s = f"{int(bpm)} BPM"
            
            elif f == 'loudness':
                # Caso 1: Valore già in dB (es. -8.5) -> Usa quello
                if val < -1.0 or val > 1.0: 
                     val_s = f"{val:.1f} dB"
                # Caso 2: Valore normalizzato -> Prova Scaler -> Fallback Formula
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

    # --- BOTTONI (Like, Dislike, Save) ---
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

            # 1. Allenamento Oracle
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

            # 2. Aggiorna Contesto
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
                
                # 1. Salva SOLO su CSV Playlist (DISCO)
                new_row = {'id': track['id'], 'name': track['name'], 'artist': track['artist'], 'genres': real_g, 'popularity': real_p, 'year': track.get('year'), **{k: track[k] for k in cols}}
                df_new = pd.DataFrame([new_row])
                df_new.to_csv(PLAYLIST_SAVED_PATH, mode='a', header=not os.path.exists(PLAYLIST_SAVED_PATH), index=False)
                
                # 2. AGGIORNA MEMORIA SESSIONE (RAM) - NON SU DISCO
                st.session_state.history_df = pd.concat([st.session_state.history_df, df_new], ignore_index=True)

                # NOTA: Non azzeriamo la coda qui, proseguiamo con la prossima
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
    
    # Recupera il vettore target
    vector_source = st.session_state.predicted_vector
    if vector_source is None and st.session_state.current_context is not None:
        vector_source = st.session_state.current_context 

    if vector_source is not None:
        # Creiamo una copia per normalizzarla SOLO per il grafico (senza toccare i dati veri)
        # Assicuriamoci che sia float per evitare errori di divisione
        plot_vec = np.array(vector_source[:9], dtype=float).copy()
        
        # --- NORMALIZZAZIONE INTELLIGENTE PER IL GRAFICO ---
        # Indici: 0=Energy, 1=Valence, 2=Dance, 3=Tempo, 4=Loud, 5=Speech, 6=Acoust, 7=Instr, 8=Live
        
        # 1. TEMPO (Index 3): Se è "reale" (es. 120), portalo a 0-1
        # Assumiamo un range tipico 40-200 BPM
        if plot_vec[3] > 2.0: 
            plot_vec[3] = (plot_vec[3] - 40) / 160.0
        
        # 2. LOUDNESS (Index 4): Se è in dB (es. -10), portalo a 0-1
        # Assumiamo un range tipico -60dB a 0dB
        if plot_vec[4] < 0:
            plot_vec[4] = (plot_vec[4] + 60) / 60.0
            
        # 3. ALTRE FEATURE (0,1,2,5,6,7,8): Se sono in scala 0-100, dividi per 100
        # Se sono già 0-1, restano uguali.
        for i in [0, 1, 2, 5, 6, 7, 8]:
            if plot_vec[i] > 1.0:
                plot_vec[i] = plot_vec[i] / 100.0

        # Clipping finale di sicurezza per restare nel grafico (0.0 - 1.0)
        vec = np.clip(plot_vec, 0, 1)
        
        # --- CREAZIONE GRAFICO ---
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