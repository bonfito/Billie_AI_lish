import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import requests
from dotenv import load_dotenv
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

load_dotenv()

# --- CONFIGURAZIONE PERCORSI ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(CURRENT_DIR, '..', 'data'))

# File di Output (Logica Separata)
HISTORY_FILE = os.path.join(DATA_DIR, "user_history.csv") # Solo Recent (Ultimi 50) - QUESTO USA L'APP
SHORT_FILE = os.path.join(DATA_DIR, "history_short.csv")  # Backup
MEDIUM_FILE = os.path.join(DATA_DIR, "history_medium.csv") # Backup
LONG_FILE = os.path.join(DATA_DIR, "history_long.csv")     # Backup

# File di Supporto
CACHE_FILE = os.path.join(DATA_DIR, "audio_features_cache.csv")
SCALER_FILE = os.path.join(DATA_DIR, "scaler.save")
LOCAL_DB_FILE = os.path.join(DATA_DIR, "tracks_db.csv")

# --- HELPER FUNCTIONS ---
def load_audio_cache():
    if not os.path.exists(CACHE_FILE): return {}
    try:
        df = pd.read_csv(CACHE_FILE)
        df = df.drop_duplicates(subset=['id'])
        return df.set_index('id').to_dict('index')
    except Exception as e:
        print(f"Errore cache: {e}")
        return {}

def save_to_cache(new_data_list):
    if not new_data_list: return
    df_new = pd.DataFrame(new_data_list)
    if not os.path.exists(CACHE_FILE):
        df_new.to_csv(CACHE_FILE, index=False)
    else:
        df_new.to_csv(CACHE_FILE, mode='a', header=False, index=False)
    print(f"Cache aggiornata: +{len(new_data_list)} brani.")

def get_reccobeats_track_info(spotify_ids):
    if not spotify_ids: return {}
    ids_str = ','.join(spotify_ids)
    url = f"https://api.reccobeats.com/v1/track?ids={ids_str}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200: return {}
        data = response.json()
        mapping = {}
        for track in data.get('content', []):
            r_id = track.get('id')
            if r_id and r_id in spotify_ids:
                mapping[r_id] = r_id
        return mapping
    except: return {}

def get_audio_features(reccobeats_id):
    url = f"https://api.reccobeats.com/v1/track/{reccobeats_id}/audio-features"
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except: return None

def enrich_metadata(df, sp):
    if df.empty: return df
    print(f"Arricchimento metadati per {len(df)} brani...")
    
    unique_ids = [uid for uid in df['id'].unique() if len(str(uid)) == 22]
    t_pop_map = {}
    
    for i in range(0, len(unique_ids), 50):
        try:
            tracks = sp.tracks(unique_ids[i:i+50])['tracks']
            for t in tracks:
                if t: t_pop_map[t['id']] = t['popularity']
        except: time.sleep(1)
        
    df['popularity'] = df['id'].map(t_pop_map).fillna(0).astype(int)
    return df

def process_and_save_list(sp, track_list, filename, source_name):
    if not track_list:
        print(f"[{source_name}] Nessun dato.")
        return

    print(f"[{source_name}] Elaborazione {len(track_list)} brani...")

    raw_data = []
    for item in track_list:
        track = item['track'] if 'track' in item else item
        if not track: continue
        
        entry = {
            'id': track['id'],
            'name': track['name'],
            'artist': track['artists'][0]['name'],
            'played_at': item.get('played_at'), 
            'source': source_name
        }
        raw_data.append(entry)

    df = pd.DataFrame(raw_data)
    
    cache = load_audio_cache()
    to_fetch = []
    final_rows = []
    new_cache_entries = []
    
    feature_keys = ['energy', 'valence', 'danceability', 'tempo', 'loudness', 
                    'speechiness', 'acousticness', 'instrumentalness', 'liveness']

    for idx, row in df.iterrows():
        tid = row['id']
        if tid in cache:
            row_dict = row.to_dict()
            row_dict.update(cache[tid])
            final_rows.append(row_dict)
        else:
            to_fetch.append(row.to_dict())

    if to_fetch:
        print(f"[{source_name}] Scaricamento features mancanti per {len(to_fetch)} brani...")
        ids = [x['id'] for x in to_fetch]
        mappings = {}
        for i in range(0, len(ids), 20):
            batch = ids[i:i+20]
            mappings.update(get_reccobeats_track_info(batch))
            time.sleep(0.2)
            
        for row in to_fetch:
            tid = row['id']
            feats = None
            if tid in mappings:
                feats = get_audio_features(mappings[tid])
            
            if feats:
                feats.pop('id', None)
                row.update(feats)
                ce = {'id': tid}
                for k in feature_keys: ce[k] = feats.get(k)
                new_cache_entries.append(ce)
            
            final_rows.append(row)
            time.sleep(0.05)
            
    if new_cache_entries:
        save_to_cache(new_cache_entries)

    df_final = pd.DataFrame(final_rows)
    df_final = enrich_metadata(df_final, sp)

    for c in feature_keys:
        if c not in df_final.columns: df_final[c] = np.nan
    df_final[feature_keys] = df_final[feature_keys].fillna(0.5)

    if os.path.exists(SCALER_FILE):
        scaler = joblib.load(SCALER_FILE)
    else:
        scaler = MinMaxScaler()
        min_d = {'energy':0,'valence':0,'danceability':0,'tempo':0,'loudness':-60,'speechiness':0,'acousticness':0,'instrumentalness':0,'liveness':0}
        max_d = {'energy':1,'valence':1,'danceability':1,'tempo':250,'loudness':0,'speechiness':1,'acousticness':1,'instrumentalness':1,'liveness':1}
        ref = pd.DataFrame([min_d, max_d])
        scaler.fit(ref[feature_keys])
        joblib.dump(scaler, SCALER_FILE)
    
    df_final[feature_keys] = scaler.transform(df_final[feature_keys])
    
    if source_name == "recent" and 'played_at' in df_final.columns:
        df_final = df_final.sort_values(by='played_at', ascending=False)
        
    df_final.to_csv(filename, index=False)
    print(f"[{source_name}] Salvato {len(df_final)} brani in {os.path.basename(filename)}")


# --- FUNZIONE PRINCIPALE (RINOMINATA PER FIX IMPORT) ---
def fetch_history():
    """
    Scarica Recently Played (per l'app) e Top Tracks (per backup separati).
    Mantiene la logica a 4 file ma espone il nome 'fetch_history' richiesto da app.py.
    """
    scope = "user-read-recently-played user-top-read"
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        scope=scope,
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI")
    ))
    
    print("=== AVVIO SCARICAMENTO DATI UTENTE ===")
    
    # 1. RECENT (User History Pura - Ultimi 50 - FILE PRINCIPALE APP)
    try:
        recent = sp.current_user_recently_played(limit=50)
        process_and_save_list(sp, recent['items'], HISTORY_FILE, "recent")
    except Exception as e:
        print(f"Errore Recent: {e}")

    # 2. SHORT TERM (Top Tracks 4 settimane - Backup)
    try:
        short = sp.current_user_top_tracks(limit=50, time_range='short_term')
        process_and_save_list(sp, short['items'], SHORT_FILE, "short_term")
    except: pass
    
    # 3. MEDIUM TERM (Top Tracks 6 mesi - Backup)
    try:
        medium = sp.current_user_top_tracks(limit=50, time_range='medium_term')
        process_and_save_list(sp, medium['items'], MEDIUM_FILE, "medium_term")
    except: pass
        
    # 4. LONG TERM (Top Tracks Anni - Backup)
    try:
        long = sp.current_user_top_tracks(limit=50, time_range='long_term')
        process_and_save_list(sp, long['items'], LONG_FILE, "long_term")
    except: pass

    print("=== OPERAZIONE COMPLETATA ===")

if __name__ == "__main__":
    fetch_history()