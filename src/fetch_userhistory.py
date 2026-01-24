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

# configurazione cache
CACHE_FILE = os.path.join("data", "audio_features_cache.csv")

def load_audio_cache():
    # Carica la cache delle audio features (valori grezzi)
    if not os.path.exists(CACHE_FILE):
        return {}
    
    try:
        df = pd.read_csv(CACHE_FILE)
        df = df.drop_duplicates(subset=['id'])
        return df.set_index('id').to_dict('index')
    except Exception as e:
        print(f"Errore lettura cache: {e}")
        return {}

def save_to_cache(new_data_list):
    # Salva i nuovi dati grezzi nella cache
    if not new_data_list:
        return

    df_new = pd.DataFrame(new_data_list)
    
    if not os.path.exists(CACHE_FILE):
        df_new.to_csv(CACHE_FILE, index=False)
    else:
        df_new.to_csv(CACHE_FILE, mode='a', header=False, index=False)
    
    print(f"Cache aggiornata: aggiunte {len(new_data_list)} nuove canzoni.")



def load_local_db_map():
    db_path = os.path.join("data", "tracks_db.csv")
    if not os.path.exists(db_path):
        print("Attenzione: file tracks_db.csv non trovato.")
        return {}
    
    print("Caricamento indice database locale...")
    try:
        df = pd.read_csv(db_path, usecols=['id', 'genre', 'popularity'], dtype={'id': str})
        df = df.drop_duplicates(subset=['id'])
        df['id'] = df['id'].str.strip()
        db_map = df.set_index('id').to_dict('index')
        return db_map
    except Exception as e:
        print(f"Errore caricamento DB locale: {e}")
        return {}

def get_reccobeats_track_info(spotify_ids):
    ids_str = ','.join(spotify_ids)
    url = f"https://api.reccobeats.com/v1/track?ids={ids_str}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200: return {}
        data = response.json()
        mapping = {}
        items = data.get('content', [])
        for track in items:
            href = track.get('href', '')
            if '/track/' in href:
                spotify_id = href.split('/track/')[-1]
                reccobeats_id = track.get('id')
                if reccobeats_id: mapping[spotify_id] = reccobeats_id
        return mapping
    except Exception:
        return {}

def get_audio_features(reccobeats_id):
    url = f"https://api.reccobeats.com/v1/track/{reccobeats_id}/audio-features"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200: return response.json()
        return None
    except Exception:
        return None


def fetch_history():
    # preaparazione dei dati cache
    local_db_map = load_local_db_map()
    audio_cache_map = load_audio_cache()
    print(f"Cache Audio caricata: {len(audio_cache_map)} canzoni in memoria.")

    #prende cronologia spotify
    scope = "user-read-recently-played"
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        scope=scope,
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI")
    ))

    print("Scaricamento cronologia recente Spotify...")
    try:
        results = sp.current_user_recently_played(limit=50)
    except Exception:
        return
    
    if not results['items']: return

    
    tracks_to_fetch_from_recco = []
    final_tracks_data = []
    count_from_cache = 0
    
    for item in results['items']:
        track = item['track']
        t_id = track['id']
        
        # Recupero dati statici (Genere/Popolarità) dal DB locale
        local_info = local_db_map.get(t_id, {'genre': 'unknown', 'popularity': 0})
        
        track_obj = {
            'id': t_id,
            'name': track['name'],
            'artist': track['artists'][0]['name'],
            'genres': local_info.get('genre', 'unknown'),
            'popularity': local_info.get('popularity', 0),
            'played_at': item['played_at']
        }

        # controllo delle feature nella cache
        if t_id in audio_cache_map:
            cached_features = audio_cache_map[t_id]
            track_obj.update(cached_features)
            track_obj['source'] = 'cache_local'
            final_tracks_data.append(track_obj)
            count_from_cache += 1
        else:
            tracks_to_fetch_from_recco.append(track_obj)

    print(f"Stato: {count_from_cache} da cache locale, {len(tracks_to_fetch_from_recco)} da scaricare.")

    # download dei mancanti da recco
    new_features_to_cache = []
    
    if tracks_to_fetch_from_recco:
        ids_to_search = [t['id'] for t in tracks_to_fetch_from_recco]
        all_mappings = {}
        batch_size = 20
        
        # mappo gli id
        for i in range(0, len(ids_to_search), batch_size):
            batch = ids_to_search[i:i+batch_size]
            mapping = get_reccobeats_track_info(batch)
            all_mappings.update(mapping)
            time.sleep(0.2)
        
        # download features
        feature_keys = ['energy', 'valence', 'danceability', 'tempo', 'loudness', 
                        'speechiness', 'acousticness', 'instrumentalness', 'liveness']

        for track_info in tracks_to_fetch_from_recco:
            sp_id = track_info['id']
            features_found = False
            
            if sp_id in all_mappings:
                recco_id = all_mappings[sp_id]
                features = get_audio_features(recco_id)
                
                if features:
                    track_info.update(features)
                    track_info['source'] = 'reccobeats'
                    features_found = True
                    
                    # preparo oggetto per cache
                    cache_entry = {'id': sp_id}
                    for k in feature_keys:
                        cache_entry[k] = features.get(k)
                    new_features_to_cache.append(cache_entry)

            if not features_found:
                track_info['source'] = 'features_missing'
            
            final_tracks_data.append(track_info)
            time.sleep(0.05)

    # salvataggio in cache
    if new_features_to_cache:
        save_to_cache(new_features_to_cache)

    # normalizzazione
    df = pd.DataFrame(final_tracks_data)
    
    audio_cols = ['energy', 'valence', 'danceability', 'tempo', 'loudness', 
                  'speechiness', 'acousticness', 'instrumentalness', 'liveness']
    
    # Gestione missing values
    for col in audio_cols:
        if col not in df.columns: df[col] = np.nan
    df[audio_cols] = df[audio_cols].fillna(df[audio_cols].mean(numeric_only=True)).fillna(0.5)

    # --- NORMALIZZAZIONE DIRETTA NEL DATAFRAME ---
    print("Normalizzazione dati (0-1)...")
    
    # Definizione limiti fisici per lo scaler
    min_data = {
        'energy': 0.0, 'valence': 0.0, 'danceability': 0.0,
        'tempo': 0.0, 'loudness': -60.0,
        'speechiness': 0.0, 'acousticness': 0.0, 'instrumentalness': 0.0, 'liveness': 0.0
    }
    max_data = {
        'energy': 1.0, 'valence': 1.0, 'danceability': 1.0,
        'tempo': 250.0, 'loudness': 0.0,
        'speechiness': 1.0, 'acousticness': 1.0, 'instrumentalness': 1.0, 'liveness': 1.0
    }
    
    ref_df = pd.DataFrame([min_data, max_data])
    scaler = MinMaxScaler()
    scaler.fit(ref_df[audio_cols])
    
    # Salviamo lo scaler per usarlo dopo (importante!)
    joblib.dump(scaler, os.path.join("data", "scaler.save"))

    # Sovrascriviamo le colonne grezze con quelle normalizzate
    df[audio_cols] = scaler.transform(df[audio_cols])

    # 7. Salvataggio Unico File
    output_path = os.path.join("data", "user_history.csv")
    
    # Definiamo l'ordine delle colonne (Metadati + Audio Features Normalizzate)
    final_cols = ['id', 'name', 'artist', 'genres', 'popularity', 'played_at', 'source'] + audio_cols
    cols_to_save = [c for c in final_cols if c in df.columns]
    
    df[cols_to_save].to_csv(output_path, index=False)
    
    print("-" * 40)
    print(f"Totale brani: {len(df)}")
    print("-" * 40)

if __name__ == "__main__":
    fetch_history()