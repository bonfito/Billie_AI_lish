import os
import time
import requests
import numpy as np
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
import joblib

load_dotenv()

# Configurazione percorsi
CACHE_FILE = os.path.join("data", "audio_features_cache.csv")
SAVED_FILE = os.path.join("data", "saved_tracks.csv")
SCALER_FILE = os.path.join("data", "scaler.save")
LOCAL_DB_FILE = os.path.join("data", "tracks_db.csv")

# Colonne audio (stessa lista usata in fetch_userhistory.py)
AUDIO_COLS = [
    'energy', 'valence', 'danceability', 'tempo', 'loudness',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness'
]


def load_audio_cache():
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
    if not new_data_list:
        return

    df_new = pd.DataFrame(new_data_list)

    if not os.path.exists(CACHE_FILE):
        df_new.to_csv(CACHE_FILE, index=False)
    else:
        df_new.to_csv(CACHE_FILE, mode='a', header=False, index=False)

    print(f"Cache aggiornata: aggiunte {len(new_data_list)} nuove canzoni.")


def load_local_db_map():
    if not os.path.exists(LOCAL_DB_FILE):
        print("Attenzione: file tracks_db.csv non trovato.")
        return {}

    print("Caricamento indice database locale...")
    try:
        df = pd.read_csv(LOCAL_DB_FILE, usecols=['id', 'genre', 'popularity'], dtype={'id': str})
        df = df.drop_duplicates(subset=['id'])
        df['id'] = df['id'].str.strip()
        db_map = df.set_index('id').to_dict('index')
        return db_map
    except Exception as e:
        print(f"Errore caricamento DB locale: {e}")
        return {}


def get_reccobeats_track_info(spotify_ids):
    if not spotify_ids:
        return {}
    ids_str = ','.join(spotify_ids)
    url = f"https://api.reccobeats.com/v1/track?ids={ids_str}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return {}
        data = response.json()
        mapping = {}
        items = data.get('content', [])
        for track in items:
            href = track.get('href', '')
            if '/track/' in href:
                spotify_id = href.split('/track/')[-1]
                reccobeats_id = track.get('id')
                if reccobeats_id:
                    mapping[spotify_id] = reccobeats_id
        return mapping
    except Exception:
        return {}


def get_audio_features(reccobeats_id):
    url = f"https://api.reccobeats.com/v1/track/{reccobeats_id}/audio-features"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def enrich_metadata(df, sp):
    """
    Scarica Popolarità (Track) e Genere (Artist) per tutti gli ID nel DataFrame.
    Salva TUTTI i generi in una singola stringa separata da '|'.
    """
    if df.empty:
        return df

    print(f"🔄 Arricchimento metadati per {len(df)} brani...")

    unique_ids = df['id'].unique().tolist()
    valid_ids = [uid for uid in unique_ids if len(str(uid)) == 22 and '-' not in str(uid)]

    track_pop_map = {}
    track_artist_map = {}
    artist_ids_set = set()

    # 1. Batch Tracks (Popolarità + Artist ID)
    for i in range(0, len(valid_ids), 50):
        batch = valid_ids[i:i+50]
        try:
            tracks_info = sp.tracks(batch)
            for t in tracks_info['tracks']:
                if t:
                    t_id = t['id']
                    track_pop_map[t_id] = t.get('popularity', 0)
                    if t.get('artists'):
                        a_id = t['artists'][0]['id']
                        track_artist_map[t_id] = a_id
                        artist_ids_set.add(a_id)
        except Exception as e:
            print(f"Errore batch tracks: {e}")
            time.sleep(1)

    # 2. Batch Artists (Generi)
    artist_genre_map = {}
    artist_ids_list = list(artist_ids_set)

    for i in range(0, len(artist_ids_list), 50):
        batch = artist_ids_list[i:i+50]
        try:
            artists_info = sp.artists(batch)
            for a in artists_info['artists']:
                if a:
                    genres = a.get('genres', []) or []
                    genre_val = "|".join(genres) if genres else 'unknown'
                    artist_genre_map[a['id']] = genre_val
        except Exception as e:
            print(f"Errore batch artists: {e}")
            time.sleep(1)

    def get_genre(t_id):
        a_id = track_artist_map.get(t_id)
        if a_id:
            return artist_genre_map.get(a_id, 'unknown')
        return 'unknown'

    new_pops = df['id'].map(track_pop_map)
    new_genres = df['id'].map(get_genre)

    if 'popularity' not in df.columns:
        df['popularity'] = 0
    if 'genres' not in df.columns:
        df['genres'] = 'unknown'

    df['popularity'] = new_pops.fillna(df['popularity']).fillna(0).astype(int)
    df['genres'] = new_genres.fillna(df['genres']).fillna('unknown')

    return df


def fetch_saved_tracks():
    # 1. Caricamento dati esistenti
    local_db_map = load_local_db_map()
    audio_cache_map = load_audio_cache()
    print(f"Cache Audio caricata: {len(audio_cache_map)} canzoni in memoria.")

    df_existing = pd.DataFrame()

    # 2. Connessione Spotify (Saved Tracks)
    scope = "user-library-read"
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        scope=scope,
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI")
    ))

    if os.path.exists(SAVED_FILE):
        try:
            df_existing = pd.read_csv(SAVED_FILE)
            if not df_existing.empty:
                print(f"Saved_tracks esistente trovato: {len(df_existing)} brani.")
                # Retroactive update metadati
                df_existing = enrich_metadata(df_existing, sp)
        except Exception as e:
            print(f"Errore lettura saved_tracks esistente: {e}")

    # ID già presenti per evitare duplicati
    existing_ids = set(df_existing['id'].astype(str).tolist()) if (not df_existing.empty and 'id' in df_existing.columns) else set()

    print("Scaricamento Saved Tracks Spotify...")

    all_items = []
    limit = 50
    offset = 0

    while True:
        try:
            page = sp.current_user_saved_tracks(limit=limit, offset=offset)
        except Exception as e:
            print(f"Errore connessione Spotify (saved tracks): {e}")
            break

        items = page.get('items', []) or []
        all_items.extend(items)

        if not page.get('next'):
            break
        offset += limit
        time.sleep(0.1)

    if not all_items:
        print("Nessun brano salvato trovato su Spotify.")
        if not df_existing.empty:
            df_existing.to_csv(SAVED_FILE, index=False)
            print("Metadati aggiornati salvati.")
        return

    # 3. Preparazione tracce da processare (solo nuove)
    tracks_to_process = []

    for item in all_items:
        track = (item or {}).get('track')
        if not track or not track.get('id'):
            continue

        t_id = str(track['id'])
        if t_id in existing_ids:
            continue

        added_at = (item or {}).get('added_at')  # ISO timestamp

        track_obj = {
            'id': t_id,
            'name': track.get('name', 'unknown'),
            'artist': (track.get('artists') or [{}])[0].get('name', 'unknown'),
            'genres': 'unknown',
            'popularity': 0,
            # Per mantenere la stessa forma di user_history.csv usiamo la colonna played_at
            'played_at': added_at,
        }
        tracks_to_process.append(track_obj)

    if not tracks_to_process:
        print("Nessun nuovo brano salvato da aggiungere.")
        if not df_existing.empty:
            df_existing.to_csv(SAVED_FILE, index=False)
            print("Metadati aggiornati salvati.")
        return

    print(f"Trovati {len(tracks_to_process)} nuovi brani salvati da elaborare.")

    # 4. Arricchimento dati (Cache vs Reccobeats)
    final_new_tracks = []
    tracks_to_fetch_from_recco = []
    new_features_to_cache = []

    for track_obj in tracks_to_process:
        t_id = track_obj['id']

        if t_id in audio_cache_map:
            cached_features = audio_cache_map[t_id]
            track_obj.update(cached_features)
            track_obj['source'] = 'cache_local'
            final_new_tracks.append(track_obj)
        else:
            tracks_to_fetch_from_recco.append(track_obj)

    # Download mancanti da Reccobeats
    if tracks_to_fetch_from_recco:
        print(f"Scaricamento features per {len(tracks_to_fetch_from_recco)} brani da API esterna...")
        ids_to_search = [t['id'] for t in tracks_to_fetch_from_recco]
        all_mappings = {}
        batch_size = 20

        for i in range(0, len(ids_to_search), batch_size):
            batch = ids_to_search[i:i+batch_size]
            mapping = get_reccobeats_track_info(batch)
            all_mappings.update(mapping)
            time.sleep(0.2)

        feature_keys = AUDIO_COLS

        for track_info in tracks_to_fetch_from_recco:
            sp_id = track_info['id']
            features_found = False

            if sp_id in all_mappings:
                recco_id = all_mappings[sp_id]
                features = get_audio_features(recco_id)

                if features:
                    features.pop('id', None)
                    track_info.update(features)
                    track_info['source'] = 'reccobeats'
                    features_found = True

                    cache_entry = {'id': sp_id}
                    for k in feature_keys:
                        cache_entry[k] = features.get(k)
                    new_features_to_cache.append(cache_entry)

            if not features_found:
                track_info['source'] = 'features_missing'

            final_new_tracks.append(track_info)
            time.sleep(0.05)

    if new_features_to_cache:
        save_to_cache(new_features_to_cache)

    if not final_new_tracks:
        if not df_existing.empty:
            df_existing.to_csv(SAVED_FILE, index=False)
        return

    # Create DF New
    df_new = pd.DataFrame(final_new_tracks)

    # Enrich metadata (popolarità + generi) anche per i nuovi brani
    df_new = enrich_metadata(df_new, sp)

    # 5. Normalizzazione (come user_history)
    for col in AUDIO_COLS:
        if col not in df_new.columns:
            df_new[col] = np.nan

    df_new[AUDIO_COLS] = df_new[AUDIO_COLS].fillna(df_new[AUDIO_COLS].mean(numeric_only=True)).fillna(0.5)

    print("Normalizzazione nuovi dati...")

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
    scaler.fit(ref_df[AUDIO_COLS])

    joblib.dump(scaler, SCALER_FILE)
    df_new[AUDIO_COLS] = scaler.transform(df_new[AUDIO_COLS])

    # 6. Merge
    if not df_existing.empty:
        df_updated = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_updated = df_new

    # Sort & Cut
    if 'played_at' in df_updated.columns:
        df_updated['played_at'] = pd.to_datetime(df_updated['played_at'], errors='coerce')
        df_updated = df_updated.sort_values(by='played_at', ascending=True)

    if len(df_updated) > 5000:
        print(f"Taglio saved_tracks: da {len(df_updated)} mantenuti ultimi 5000.")
        df_updated = df_updated.tail(5000)

    # 7. Save (stessa forma di user_history.csv)
    final_cols = ['id', 'name', 'artist', 'genres', 'popularity', 'played_at', 'source'] + AUDIO_COLS
    for c in final_cols:
        if c not in df_updated.columns:
            df_updated[c] = np.nan

    df_updated[final_cols].to_csv(SAVED_FILE, index=False)

    print("-" * 40)
    print(f"Operazione completata. Totale brani salvati: {len(df_updated)}")
    print(f"Salvato in: {SAVED_FILE}")
    print("-" * 40)


if __name__ == "__main__":
    fetch_saved_tracks()