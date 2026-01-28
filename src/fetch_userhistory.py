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

# Configurazione percorsi
CACHE_FILE = os.path.join("data", "audio_features_cache.csv")
HISTORY_FILE = os.path.join("data", "user_history.csv")
SCALER_FILE = os.path.join("data", "scaler.save")
LOCAL_DB_FILE = os.path.join("data", "tracks_db.csv")

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

def enrich_metadata(df, sp):
    """
    Scarica Popolarità (Track) e Genere (Artist) per tutti gli ID nel DataFrame.
    """
    if df.empty:
        return df

    print(f"🔄 Arricchimento metadati per {len(df)} brani...")
    
    # Filtriamo ID validi (Spotify IDs sono 22 char, no UUID)
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
                    track_pop_map[t_id] = t['popularity']
                    if t['artists']:
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
                    genres = a.get('genres', [])
                    # Prendiamo il primo genere o 'unknown'
                    genre_val = genres[0] if genres else 'unknown'
                    artist_genre_map[a['id']] = genre_val
        except Exception as e:
            print(f"Errore batch artists: {e}")
            time.sleep(1)

    # 3. Aggiornamento DataFrame
    # Funzione helper per mappare
    def get_genre(t_id):
        a_id = track_artist_map.get(t_id)
        if a_id:
            return artist_genre_map.get(a_id, 'unknown')
        return 'unknown'

    # Applichiamo solo dove abbiamo dati (combine_first o update diretto)
    # Creiamo le serie mappate
    new_pops = df['id'].map(track_pop_map)
    new_genres = df['id'].map(get_genre)
    
    # Aggiorniamo le colonne esistenti, se i valori sono NaN nel DF originale o vogliamo sovrascrivere?
    # L'utente vuole che SI PRENDANO i dati, quindi sovrascriviamo per avere dati freschi.
    df['popularity'] = new_pops.fillna(df['popularity']).fillna(0).astype(int)
    
    # Per i generi, sovrascriviamo se non è 'unknown' quello che abbiamo trovato
    # O più semplicemente: sovrascriviamo tutto ciò che abbiamo trovato dalla mappa.
    # Se la mappa da 'unknown' (perché artista senza genere), pazienza.
    df['genres'] = new_genres.fillna(df['genres']).fillna('unknown')
    
    return df

def fetch_history():
    # 1. Caricamento dati esistenti
    local_db_map = load_local_db_map()
    audio_cache_map = load_audio_cache()
    print(f"Cache Audio caricata: {len(audio_cache_map)} canzoni in memoria.")

    # 2. Connessione Spotify (Aggiornato scope per Top Tracks)
    scope = "user-read-recently-played user-top-read"
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        scope=scope,
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI")
    ))

    print("Scaricamento dati Spotify (Recenti + Top Tracks)...")
    
    # --- LOGICA MULTI-SORGENTE E PESI ---
    # Usiamo un dizionario per aggregare i brani ed evitare duplicati
    raw_tracks_map = {} 

    def process_spotify_items(items, source_label, weight_increment, is_history=False):
        for item in items:
            # In history l'oggetto track è dentro 'track', in top tracks è l'oggetto stesso
            track = item['track'] if is_history else item
            if not track: continue
            
            t_id = track['id']
            played_at = pd.to_datetime(item['played_at']) if is_history else None
            
            # Se è la prima volta che vediamo il brano
            if t_id not in raw_tracks_map:
                # Recuperiamo info base (se disponibili in locale)
                local_info = local_db_map.get(t_id, {'genre': 'unknown', 'popularity': 0})
                
                raw_tracks_map[t_id] = {
                    'id': t_id,
                    'name': track['name'],
                    'artist': track['artists'][0]['name'],
                    'genres': local_info.get('genre', 'unknown'),
                    'popularity': local_info.get('popularity', 0),
                    'played_at': played_at,
                    'weight': 0.0,
                    'sources': set()
                }
            
            # Aggiorniamo peso e sorgenti
            entry = raw_tracks_map[t_id]
            entry['weight'] += weight_increment
            entry['sources'].add(source_label)
            
            # Aggiorniamo il played_at se ne troviamo uno più recente (o se quello attuale è None)
            if played_at:
                if entry['played_at'] is None or played_at > entry['played_at']:
                    entry['played_at'] = played_at

    try:
        # A. Recent History (Peso 2.0 - Mood Attuale)
        results_recent = sp.current_user_recently_played(limit=50)
        process_spotify_items(results_recent['items'], 'recent', 2.0, is_history=True)

        # B. Short Term (4 settimane - Peso 1.5 - Ossessione Corrente)
        results_short = sp.current_user_top_tracks(limit=50, time_range='short_term')
        process_spotify_items(results_short['items'], 'short_term', 1.5)

        # C. Medium Term (6 mesi - Peso 1.0 - Gusto Consolidato)
        results_medium = sp.current_user_top_tracks(limit=50, time_range='medium_term')
        process_spotify_items(results_medium['items'], 'medium_term', 1.0)

        # D. Long Term (Anni - Peso 0.5 - Identità Storica)
        results_long = sp.current_user_top_tracks(limit=50, time_range='long_term')
        process_spotify_items(results_long['items'], 'long_term', 0.5)

    except Exception as e:
        print(f"Errore chiamate API Spotify: {e}")
        return

    # Convertiamo il dizionario in lista
    tracks_to_process = list(raw_tracks_map.values())
    
    # Formattiamo il campo 'sources' come stringa per il CSV
    for t in tracks_to_process:
        t['sources'] = ",".join(sorted(list(t['sources'])))

    if not tracks_to_process:
        print("Nessun dato trovato da elaborare.")
        return

    print(f"Totale brani unici identificati: {len(tracks_to_process)}")

    # 4. Arricchimento dati (Cache vs Reccobeats)
    final_new_tracks = []
    tracks_to_fetch_from_recco = []
    new_features_to_cache = []

    for track_obj in tracks_to_process:
        t_id = track_obj['id']
        
        if t_id in audio_cache_map:
            cached_features = audio_cache_map[t_id]
            track_obj.update(cached_features)
            # Nota: 'source' qui indicava l'origine dei dati tecnici, possiamo lasciarlo o sovrascriverlo.
            # Nel tuo codice originale era 'cache_local'. Manteniamo la logica originale se serve debug, 
            # ma abbiamo già la colonna 'sources' (es. recent,short_term) che è più importante per l'AI.
            # Per non rompere nulla, lascio che track_obj abbia i dati cache.
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
        
        feature_keys = ['energy', 'valence', 'danceability', 'tempo', 'loudness', 
                        'speechiness', 'acousticness', 'instrumentalness', 'liveness']

        for track_info in tracks_to_fetch_from_recco:
            sp_id = track_info['id']
            features_found = False
            
            if sp_id in all_mappings:
                recco_id = all_mappings[sp_id]
                features = get_audio_features(recco_id)
                
                if features:
                    # --- FIX PROTEZIONE ID ---
                    features.pop('id', None)
                    # -------------------------

                    track_info.update(features)
                    # track_info['source'] = 'reccobeats' # Opzionale, come da codice originale
                    features_found = True
                    
                    cache_entry = {'id': sp_id}
                    for k in feature_keys:
                        cache_entry[k] = features.get(k)
                    new_features_to_cache.append(cache_entry)

            if not features_found:
                # track_info['source'] = 'features_missing'
                pass
            
            final_new_tracks.append(track_info)
            time.sleep(0.05)

    if new_features_to_cache:
        save_to_cache(new_features_to_cache)

    if not final_new_tracks:
        print("Nessun brano valido finale.")
        return

    # Create DF
    df_new = pd.DataFrame(final_new_tracks)
    
    # --- ENRICH METADATA ---
    # Scarichiamo popolarità e genere (per i brani che non li avevano in cache locale)
    df_new = enrich_metadata(df_new, sp)
    # -----------------------

    # 5. Normalizzazione
    audio_cols = ['energy', 'valence', 'danceability', 'tempo', 'loudness', 
                  'speechiness', 'acousticness', 'instrumentalness', 'liveness']
    
    for col in audio_cols:
        if col not in df_new.columns: df_new[col] = np.nan
    
    # Fill NaN con media o default
    df_new[audio_cols] = df_new[audio_cols].fillna(df_new[audio_cols].mean(numeric_only=True)).fillna(0.5)

    print("Normalizzazione dati...")
    
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
    
    joblib.dump(scaler, SCALER_FILE)
    df_new[audio_cols] = scaler.transform(df_new[audio_cols])

    # 6. Salvataggio (Snapshot del Profilo)
    # Ordiniamo per peso decrescente (i brani più importanti per il recommender in alto)
    df_new = df_new.sort_values(by='weight', ascending=False)

    final_cols = ['id', 'name', 'artist', 'genres', 'popularity', 'weight', 'sources', 'played_at'] + audio_cols
    cols_to_save = [c for c in final_cols if c in df_new.columns]
    
    # Salviamo sovrascrivendo (o rigenerando) il file, dato che è un profilo pesato completo
    df_new[cols_to_save].to_csv(HISTORY_FILE, index=False)
    
    print("-" * 40)
    print(f"Operazione completata. Profilo utente generato con {len(df_new)} brani.")
    print("-" * 40)

if __name__ == "__main__":
    fetch_history()