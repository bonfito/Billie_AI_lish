import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import time
import joblib
from dotenv import load_dotenv

# Import dai moduli locali
from recommender import SongRecommender
from oracle import MusicOracle 
from utils import calculate_avalanche_context
from spotify_client import add_track_to_playlist, get_track_details

# --- CONFIGURAZIONE PERCORSI ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(CURRENT_DIR, '..', 'data'))
HISTORY_PATH = os.path.join(DATA_DIR, 'user_history.csv')
BLACKLIST_PATH = os.path.join(DATA_DIR, 'blacklist.txt')
SCALER_PATH = os.path.join(DATA_DIR, 'scaler.save')

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

# --- CSS STABILE ---
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
        border-radius: 15px;
        border: 1px solid #333;
        height: 352px; /* Stessa altezza del player */
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
        height: 450px !important; 
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

# --- FUNZIONE GENERAZIONE (CORRETTA) ---
def generate_new_recommendation():
    # Rimosso argomento manual_target
    if st.session_state.history_df is not None:
        try:
            # Rimosso target_features=... dalla chiamata
            recs_df, pred_vector = st.session_state.recommender.recommend(
                st.session_state.history_df, 
                k=20
            )
            
            best_song = recs_df.iloc[0]
            st.session_state.current_track = best_song.to_dict()
            st.session_state.predicted_vector = pred_vector.flatten()
            st.session_state.suggestion_made = True
            st.session_state.past_track_ids.append(str(best_song['id']))
            return True
        except Exception as e:
            st.error(f"Errore generazione: {e}")
            return False
    return False

# --- 1. INIZIALIZZAZIONE ---
@st.cache_resource(show_spinner="Caricamento Motore AI...")
def load_engines():
    """Carica i modelli pesanti una sola volta e li tiene in RAM."""
    oracle = MusicOracle()
    try:
        recommender = SongRecommender()
    except Exception as e:
        return None, None, e
    return oracle, recommender, None

# Inizializzazione Sessione
if 'recommender' not in st.session_state:
    oracle, rec, err = load_engines()
    
    if err:
        st.error(f"Errore critico avvio: {err}")
        st.stop()
    
    st.session_state.oracle = oracle
    st.session_state.recommender = rec
    
    # Variabili di stato leggere
    st.session_state.past_track_ids = []
    st.session_state.suggestion_made = False
    st.session_state.current_track = None
    st.session_state.predicted_vector = None
    
@st.cache_data(ttl=5) 
def load_history_data(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# --- 2. CARICAMENTO STORIA ---
if 'history_df' not in st.session_state:
    try:
        if os.path.exists(HISTORY_PATH):
            history_df = pd.read_csv(HISTORY_PATH)
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
        else: raise FileNotFoundError()
    except:
        st.session_state.history_df = None
        st.session_state.current_context = np.array([0.5] * 9)
        st.session_state.song_count = 0
        st.session_state.top_artist = "-"
        st.session_state.top_genre = "-"

# --- HEADER ---
st.markdown("<div class='main-title'>BILLIE AI-LISH</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Artificial Music Agent</div>", unsafe_allow_html=True)

# --- SIDEBAR (LOGICA: SLIDER 0-100 & SCALER) ---
st.sidebar.header("CONTROL ROOM")
st.sidebar.caption(f"Genre: {st.session_state.get('top_genre', '-')} | Artist: {st.session_state.get('top_artist', '-')}")
st.sidebar.markdown("---")

if st.session_state.get('current_context') is not None:
    with st.sidebar.form("dna_equalizer"):
        st.markdown("### 🧬 DNA Equalizer")
        
        ctx = st.session_state.current_context
        
        # Denormalizzazione per slider (se scaler esiste)
        start_vals = ctx
        if scaler:
            try:
                start_vals = scaler.inverse_transform(ctx.reshape(1, -1))[0]
            except:
                pass

        st.markdown("**VIBE**")
        n_en = st.slider("Energy", 0, 100, int(np.clip(ctx[0], 0, 1) * 100))
        n_val = st.slider("Valence (Mood)", 0, 100, int(np.clip(ctx[1], 0, 1) * 100))
        n_dan = st.slider("Danceability", 0, 100, int(np.clip(ctx[2], 0, 1) * 100))
        
        st.markdown("**SOUND**")
        # Default presi dai valori reali (BPM, dB)
        def_tem = float(np.clip(start_vals[3], 40, 200)) if scaler else float(np.clip(ctx[3], 40, 200))
        def_lou = float(np.clip(start_vals[4], -60, 0)) if scaler else float(np.clip(ctx[4], -60, 0))
        
        n_tem = st.slider("Tempo (BPM)", 40.0, 200.0, def_tem) 
        n_lou = st.slider("Loudness (dB)", -60.0, 0.0, def_lou)
        
        st.markdown("**TEXTURE**")
        n_spe = st.slider("Speechiness", 0.0, 1.0, float(np.clip(ctx[5], 0, 1)))
        n_aco = st.slider("Acousticness", 0, 100, int(np.clip(ctx[6], 0, 1) * 100))
        n_ins = st.slider("Instrumentalness", 0, 100, int(np.clip(ctx[7], 0, 1) * 100))
        n_liv = st.slider("Liveness", 0, 100, int(np.clip(ctx[8], 0, 1) * 100))
        
        submitted = st.form_submit_button("APPLICA & RIGENERA")
        
        if submitted:
            # 1. Normalizzazione Input Slider -> 0-1
            raw_target = [
                n_en / 100.0, n_val / 100.0, n_dan / 100.0,
                n_tem, n_lou, n_spe,
                n_aco / 100.0, n_ins / 100.0, n_liv / 100.0
            ]
            
            final_target_norm = raw_target
            # 2. Se abbiamo lo scaler, trasformiamo i valori reali (es. 120BPM) in valori AI (0.5)
            if scaler:
                try:
                    final_target_norm = scaler.transform([raw_target])[0]
                except:
                    pass

            # 3. Aggiorniamo il contesto grafico
            st.session_state.current_context = final_target_norm
            
            with st.spinner("Modulazione frequenze AI in corso..."):
                # 4. Chiamiamo senza argomenti
                if generate_new_recommendation():
                    time.sleep(0.2)
                    st.rerun()
else:
    st.sidebar.warning("Inizializza la history per attivare l'equalizzatore.")

# --- GENERAZIONE ---
c1, col_gen, c3 = st.columns([1, 2, 1])
with col_gen:
    if st.button("GENERA NUOVA VISIONE", type="primary", key="main_gen"):
        with st.spinner("L'AI sta scansionando il database..."):
            if generate_new_recommendation():
                time.sleep(0.3)
                st.rerun()

# --- DISPLAY CANZONE ---
if st.session_state.suggestion_made and st.session_state.current_track:
    track = st.session_state.current_track
    tid = track.get('id')
    
    col_player, col_stats = st.columns([2, 1]) 
    
    # 1. Colonna Sinistra: Player Spotify
    with col_player:
        if pd.notna(tid):
            url = f"https://open.spotify.com/embed/track/{tid}?utm_source=generator&theme=0"
            st.markdown(f'<div class="spotify-container"><iframe src="{url}" width="100%" height="352" frameBorder="0" allow="autoplay; encrypted-media; fullscreen; picture-in-picture"></iframe></div>', unsafe_allow_html=True)
        else:
            st.warning("Anteprima non disponibile per questa traccia.")

    # 2. Colonna Destra: Feature List
    with col_stats:
        audio_cols = ['energy', 'valence', 'danceability', 'tempo', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness']
        
        # --- PREPARAZIONE DATI REALI ---
        norm_vector = np.array([track.get(c, 0) for c in audio_cols]).reshape(1, -1)
        real_data_map = {}
        if scaler:
            try:
                real_vector = scaler.inverse_transform(norm_vector)[0]
                real_data_map = dict(zip(audio_cols, real_vector))
            except:
                pass 
        
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

    # Info Testuali
    st.markdown(f"<div class='track-name'>{track['name']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='artist-name'>{track['artist']}</div>", unsafe_allow_html=True)
    g = str(track.get('genres', 'unknown')).title()
    y = int(track.get('year', 0))
    st.markdown(f"<div class='meta-tag'>{y} • {g}</div>", unsafe_allow_html=True)

    # BOTTONI
    b1, b_save, b_skip, b4 = st.columns([1, 2, 2, 1])
    with b_save:
        if st.button("SALVA", key="btn_save"):
            with st.status("Apprendimento e Generazione...", expanded=False) as status:
                real_g, real_p = get_track_details(track['id'])
                cols = st.session_state.recommender.audio_cols
                feats = np.array([track[k] for k in cols])
                
                # 1. Allenamento
                st.session_state.oracle.train_incremental(st.session_state.current_context, feats)
                st.session_state.song_count += 1
                st.session_state.current_context = calculate_avalanche_context(st.session_state.current_context, feats, st.session_state.song_count)
                
                # 2. Spotify
                if tid: add_track_to_playlist(tid)
                
                # 3. Blacklist
                with open(BLACKLIST_PATH, "a") as f: 
                    f.write(f"{track['id']}\n")
                
                # 4. History CSV
                new_row = {'id': track['id'], 'name': track['name'], 'artist': track['artist'], 'genres': real_g, 'popularity': real_p, 'year': track.get('year'), **{k: track[k] for k in cols}}
                df_new = pd.DataFrame([new_row])
                df_new.to_csv(HISTORY_PATH, mode='a', header=not os.path.exists(HISTORY_PATH), index=False)
                st.session_state.history_df = pd.concat([st.session_state.history_df, df_new], ignore_index=True)
                
                # 5. Generazione
                generate_new_recommendation()
                status.update(label="Salvato e Blacklistato!", state="complete")
            st.rerun()

    with b_skip:
        if st.button("SKIP", key="btn_skip"):
            with st.spinner("Scartando..."):
                with open(BLACKLIST_PATH, "a") as f: 
                    f.write(f"{track['id']}\n")
                st.session_state.past_track_ids.append(str(track['id']))
                generate_new_recommendation()
                time.sleep(0.2)
                st.rerun()

st.markdown("---")

# --- HISTORY & RADAR ---
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

# Radar
st.markdown("<div style='letter-spacing: 2px; font-weight: 900; color: #444; margin-top:20px; font-size: 0.8rem;'>TARGET DNA</div>", unsafe_allow_html=True)

vector_to_plot = st.session_state.predicted_vector
if vector_to_plot is None and st.session_state.current_context is not None:
    vector_to_plot = st.session_state.current_context 

if vector_to_plot is not None:
    labels = ['Energy', 'Valence', 'Dance', 'Tempo', 'Loud', 'Speech', 'Acoust', 'Instr', 'Live']
    vec = np.clip(vector_to_plot[:9], 0, 1)
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
        height=350,
        margin=dict(l=40, r=40, t=20, b=20)
    )
    
    fig.update_traces(
        line_color='#1DB954', 
        fill='toself', 
        fillcolor='rgba(29, 185, 84, 0.15)',
        mode='lines+markers',
        marker=dict(size=6),
        hovertemplate='<b>%{theta}</b>: %{r:.2f}<extra></extra>'
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})