import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import time
from recommender import SongRecommender
from oracle import MusicOracle 
from utils import calculate_avalanche_context
from spotify_client import add_track_to_playlist, get_track_details

# --- CONFIGURAZIONE PERCORSI ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_PATH = os.path.normpath(os.path.join(CURRENT_DIR, '..', 'data', 'user_history.csv'))
BLACKLIST_PATH = os.path.normpath(os.path.join(CURRENT_DIR, '..', 'data', 'blacklist.txt'))

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(
    page_title="Billie AI-lish", 
    layout="centered", 
    page_icon=None,
    initial_sidebar_state="expanded" # Tenta di aprirla all'avvio
)

# --- CSS FIXATO (ORA LA SIDEBAR TORNA) ---
st.markdown("""
    <style>
    /* 1. Nascondiamo SOLO footer e menu hamburger, NON l'header intero */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Questo permette di vedere la freccetta '>' per riaprire la sidebar */
    header {
        visibility: visible !important;
        background: transparent !important;
    }

    /* 2. Layout Principale */
    .block-container {
        max-width: 900px !important; 
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        margin: 0 auto;
        padding-top: 2rem;
    }

    /* Centra tutto tranne la sidebar */
    section[data-testid="stMain"] [data-testid="stVerticalBlock"] > div {
        width: 100% !important;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    /* 3. Tipografia */
    .main-title { font-size: 3.5rem; font-weight: 900; color: #FFFFFF; text-transform: uppercase; margin-bottom: 0; }
    .subtitle { font-size: 1rem; letter-spacing: 5px; color: #1DB954; text-transform: uppercase; margin-bottom: 3rem; }

    .track-name { font-size: 3rem; font-weight: 800; margin-top: 1.5rem; line-height: 1.1; color: #fff; }
    .artist-name { font-size: 1.6rem; font-weight: 400; color: #1DB954; margin-bottom: 0.5rem; }
    .meta-tag { font-size: 0.9rem; color: #555; letter-spacing: 2px; margin-bottom: 2rem; text-transform: uppercase; }

    /* 4. Player */
    .spotify-container {
        width: 100%;
        display: flex;
        justify-content: center;
        margin: 0 auto;
    }
    iframe {
        border-radius: 20px;
        box-shadow: 0 20px 50px rgba(0,0,0,0.6);
    }

    /* 5. History Table */
    .history-container {
        width: 100%;
        max-width: 750px;
        max-height: 450px; 
        overflow-y: auto; 
        margin: 20px auto;
        padding: 0 15px;
        border-top: 1px solid #333;
    }
    .history-container::-webkit-scrollbar { width: 6px; }
    .history-container::-webkit-scrollbar-thumb { background: #1DB954; border-radius: 10px; }

    .history-table { width: 100%; border-collapse: collapse; font-family: 'Courier New', monospace; }
    .history-table td { padding: 12px 5px; border-bottom: 1px solid #1a1a1a; font-size: 0.85rem; color: #888; vertical-align: middle; }
    .track-number { width: 40px; color: #444; font-weight: bold; text-align: left; }
    .track-title-cell { text-align: left !important; color: #eee; letter-spacing: -0.5px; }
    .history-row-artist { color: #1DB954; font-weight: 600; opacity: 0.8; }
    .history-table tr:hover td { background-color: #111; color: #fff; }

    /* 6. Bottoni */
    section[data-testid="stMain"] div[data-testid="stButton"] { display: flex; justify-content: center; width: 100%; }
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
    
    /* 7. Sidebar Style */
    section[data-testid="stSidebar"] {
        background-color: #0e0e0e;
        border-right: 1px solid #222;
    }
    /* Forza la visibilità del bottone di toggle sidebar */
    button[kind="header"] {
        visibility: visible !important;
        color: #fff !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- FUNZIONE GENERAZIONE ---
def generate_new_recommendation():
    if st.session_state.history_df is not None:
        try:
            recs_df, pred_vector = st.session_state.recommender.recommend(st.session_state.history_df, k=20)
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
if 'oracle' not in st.session_state:
    with st.spinner("Sintonizzando Billie AI-lish..."):
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

# --- SIDEBAR (VISIBILE) ---
st.sidebar.header("🎛️ CONTROL ROOM")
st.sidebar.caption(f"Genre: {st.session_state.get('top_genre', '-')} | Artist: {st.session_state.get('top_artist', '-')}")
st.sidebar.markdown("---")

if st.session_state.get('current_context') is not None:
    with st.sidebar.form("dna_equalizer"):
        st.markdown("### 🧬 DNA Equalizer")
        
        ctx = st.session_state.current_context
        
        st.markdown("**VIBE**")
        n_en = st.slider("Energy", 0.0, 1.0, float(np.clip(ctx[0], 0, 1)))
        n_val = st.slider("Valence (Mood)", 0.0, 1.0, float(np.clip(ctx[1], 0, 1)))
        n_dan = st.slider("Danceability", 0.0, 1.0, float(np.clip(ctx[2], 0, 1)))
        
        st.markdown("**SOUND**")
        n_tem = st.slider("Tempo (BPM)", 40.0, 200.0, float(np.clip(ctx[3], 40, 200)))
        n_lou = st.slider("Loudness (dB)", -60.0, 0.0, float(np.clip(ctx[4], -60, 0)))
        
        st.markdown("**TEXTURE**")
        n_spe = st.slider("Speechiness", 0.0, 1.0, float(np.clip(ctx[5], 0, 1)))
        n_aco = st.slider("Acousticness", 0.0, 1.0, float(np.clip(ctx[6], 0, 1)))
        n_ins = st.slider("Instrumentalness", 0.0, 1.0, float(np.clip(ctx[7], 0, 1)))
        n_liv = st.slider("Liveness", 0.0, 1.0, float(np.clip(ctx[8], 0, 1)))
        
        submitted = st.form_submit_button("APPLICA & RIGENERA")
        
        if submitted:
            new_ctx = np.array([n_en, n_val, n_dan, n_tem, n_lou, n_spe, n_aco, n_ins, n_liv])
            st.session_state.current_context = new_ctx
            with st.spinner("Modulazione frequenze AI in corso..."):
                if generate_new_recommendation():
                    time.sleep(0.2)
                    st.rerun()
else:
    st.sidebar.warning("Inizializza la history per attivare l'equalizzatore.")

# --- MAIN: GENERAZIONE ---
c1, col_gen, c3 = st.columns([1, 2, 1])
with col_gen:
    if st.button("GENERA NUOVA VISIONE", type="primary", key="main_gen"):
        with st.spinner("L'AI sta scansionando il database..."):
            if generate_new_recommendation():
                time.sleep(0.3)
                st.rerun()

# --- DISPLAY ---
if st.session_state.suggestion_made and st.session_state.current_track:
    track = st.session_state.current_track
    tid = track.get('id')
    
    if pd.notna(tid):
        url = f"https://open.spotify.com/embed/track/{tid}?utm_source=generator&theme=0"
        st.markdown(f'<div class="spotify-container"><iframe src="{url}" width="100%" height="352" frameBorder="0" allow="autoplay; encrypted-media; fullscreen; picture-in-picture"></iframe></div>', unsafe_allow_html=True)

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
                
                st.session_state.oracle.train_incremental(st.session_state.current_context, feats)
                st.session_state.song_count += 1
                st.session_state.current_context = calculate_avalanche_context(st.session_state.current_context, feats, st.session_state.song_count)
                
                if tid: add_track_to_playlist(tid)
                
                new_row = {'id': track['id'], 'name': track['name'], 'artist': track['artist'], 'genres': real_g, 'popularity': real_p, 'year': track.get('year'), **{k: track[k] for k in cols}}
                df_new = pd.DataFrame([new_row])
                df_new.to_csv(HISTORY_PATH, mode='a', header=not os.path.exists(HISTORY_PATH), index=False)
                st.session_state.history_df = pd.concat([st.session_state.history_df, df_new], ignore_index=True)
                
                generate_new_recommendation()
                status.update(label="Fatto!", state="complete")
            st.rerun()

    with b_skip:
        if st.button("SKIP", key="btn_skip"):
            with st.spinner("Scartando..."):
                with open(BLACKLIST_PATH, "a") as f: f.write(f"{track['id']}\n")
                st.session_state.past_track_ids.append(str(track['id']))
                generate_new_recommendation()
                time.sleep(0.2)
                st.rerun()

    st.markdown("---")

    # HISTORY & RADAR
    st.markdown("<div style='letter-spacing: 5px; font-weight: 900; color: #444; margin-top:20px; font-size: 0.7rem;'>LATEST DISCOVERIES</div>", unsafe_allow_html=True)
    if st.session_state.history_df is not None:
        recent = st.session_state.history_df[['name', 'artist']].tail(50).iloc[::-1].reset_index(drop=True)
        html_h = "<div class='history-container'><table class='history-table'>"
        for i, r in recent.iterrows():
            num = str(len(recent) - i).zfill(2)
            html_h += f"<tr><td class='track-number'>{num}</td><td class='track-title-cell'>{r['name']} <span class='history-row-artist'> // {r['artist']}</span></td></tr>"
        html_h += "</table></div>"
        st.markdown(html_h, unsafe_allow_html=True)

    if st.session_state.predicted_vector is not None:
        st.markdown("<div style='letter-spacing: 2px; font-weight: 900; color: #444; margin-top:20px; font-size: 0.8rem;'>TARGET DNA</div>", unsafe_allow_html=True)
        labels = ['Energy', 'Valence', 'Dance', 'Tempo', 'Loud', 'Speech', 'Acoust', 'Instr', 'Live']
        vec = np.clip(st.session_state.predicted_vector[:9], 0, 1)
        df_r = pd.DataFrame(dict(r=vec, theta=labels))
        fig = px.line_polar(df_r, r='r', theta='theta', line_close=True, range_r=[0, 1])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', polar=dict(bgcolor='rgba(0,0,0,0)', radialaxis=dict(visible=False), angularaxis=dict(color='#888')), showlegend=False, height=350)
        fig.update_traces(line_color='#1DB954', fill='toself', fillcolor='rgba(29, 185, 84, 0.15)')
        st.plotly_chart(fig, use_container_width=True)