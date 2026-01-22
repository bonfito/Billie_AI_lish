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



def get_reccobeats_track_info(spotify_ids):
    #mappa id spotify in id recco
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
    except Exception as e:
        print(f"[ERRORE] Chiamata Reccobeats fallita: {e}")
        return {}

def get_audio_features(reccobeats_id):
    #scarica audio feature per ogni singola traccia
    #viene preso id recco dalla call precedente e poi lo si usa per estrarre le audio feature
    url = f"https://api.reccobeats.com/v1/track/{reccobeats_id}/audio-features"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None



def fetch_history():

    #connessione a spotify
    scope = "user-read-recently-played"
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        scope=scope,
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI")
    ))

    print("Connesso a Spotify. Scaricamento cronologia recente...")
    results = sp.current_user_recently_played(limit=50)
    
    if not results['items']:
        print("Cronologia Spotify vuota.")
        return

    # estrazione dati
    spotify_ids = []
    tracks_data = []
    
    for item in results['items']:
        track = item['track']
        t_id = track['id']
        spotify_ids.append(t_id)
        
        tracks_data.append({
            'id': t_id,
            'name': track['name'],
            'artist': track['artists'][0]['name'],
            'played_at': item['played_at']
        })
    
    print(f"Trovate {len(spotify_ids)} tracce. Interrogo Recco Beats...")

    #avviene suddivisione in batch per effettuare la richiesta all'api
    batch_size = 20
    all_mappings = {}
    
    for i in range(0, len(spotify_ids), batch_size):
        batch = spotify_ids[i:i+batch_size]
        mapping = get_reccobeats_track_info(batch)
        all_mappings.update(mapping)
        time.sleep(0.2)
    
    print(f"Tradotti {len(all_mappings)}/{len(spotify_ids)} ID.")

    # recupero le audio features
    final_tracks = []
    features_found = 0
    
    print("scarico features audio...")
    
    for track_info in tracks_data:
        sp_id = track_info['id']
        
        # Recupero features
        if sp_id in all_mappings:
            recco_id = all_mappings[sp_id]
            features = get_audio_features(recco_id)
            
            if features:
                track_info.update(features)
                track_info['source'] = 'reccobeats'
                features_found += 1
            else:
                track_info['source'] = 'features_missing'
        else:
            track_info['source'] = 'id_missing'
        
        final_tracks.append(track_info)
        time.sleep(0.1)

    # salvataggio user history
    df = pd.DataFrame(final_tracks)
    
    required_cols = ['energy', 'valence', 'danceability', 'tempo', 'loudness', 
                     'speechiness', 'acousticness', 'instrumentalness', 'liveness']
    
    # Creazione colonne mancanti e gestione NaN
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    # fallback, nel caso in cui non vi sia un dato lo si riempe con una media dei valori (non si hanno vuoti così)
    df[required_cols] = df[required_cols].fillna(df[required_cols].mean(numeric_only=True))
    df[required_cols] = df[required_cols].fillna(0.5)
    
    # salvataggio file senza normalizzaizione
    raw_path = os.path.join("data", "user_history.csv")
    final_cols = ['id', 'name', 'artist', 'played_at', 'source'] + required_cols
    cols_to_save = [c for c in final_cols if c in df.columns]
    
    df[cols_to_save].to_csv(raw_path, index=False)
    print(f"[SUCCESS] Dati grezzi salvati in: {raw_path}")

    # fase di normalizzazione dei dati, in modo da 'allineare' il dataset alla user history
    print("Avvio normalizzazione dati...")


    #prendiamo i valori di BMP (0-250) e Loudness (-60-0) e li normalizziamo con valori tra 0 e 1
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
    
    # addestramento scaler
    ref_df = pd.DataFrame([min_data, max_data])
    scaler = MinMaxScaler()
    scaler.fit(ref_df[required_cols])
    
    # salvataggio scaler 
    joblib.dump(scaler, os.path.join("data", "scaler.save"))

    # creazione di una copia della user history
    df_processed = df.copy()
    
    # Trasformiamo i dati
    df_processed[required_cols] = scaler.transform(df[required_cols])
    
    # salvatagigo file 'processato'
    processed_path = os.path.join("data", "user_history_processed.csv")
    df_processed[cols_to_save].to_csv(processed_path, index=False)
    
    print("-" * 30)
    print(f"Dati processati (0-1) salvati in: {processed_path}")
    

if __name__ == "__main__":
    fetch_history()