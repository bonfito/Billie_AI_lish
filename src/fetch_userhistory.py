import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import requests
from dotenv import load_dotenv
import time

load_dotenv()

def get_reccobeats_track_info(spotify_ids):
    """
    Ottiene le informazioni delle tracce da Reccobeats usando gli Spotify IDs.
    Ritorna un dizionario {spotify_id: reccobeats_id}
    """
    # Converte la lista in stringa separata da virgole
    ids_str = ','.join(spotify_ids)
    url = f"https://api.reccobeats.com/v1/track?ids={ids_str}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Crea mappatura spotify_id -> reccobeats_id
        mapping = {}
        if 'content' in data:
            for track in data['content']:
                # Estrae lo Spotify ID dall'href
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
    """
    Ottiene le audio features di una traccia usando l'ID di Reccobeats.
    """
    url = f"https://api.reccobeats.com/v1/track/{reccobeats_id}/audio-features"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[WARNING] Audio features non disponibili per {reccobeats_id}: {e}")
        return None

def fetch_history():
    # 1. CONNESSIONE A SPOTIFY
    scope = "user-read-recently-played"
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        scope=scope,
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI")
    ))

    print("[INFO] Connesso a Spotify. Scaricamento cronologia recente...")
    results = sp.current_user_recently_played(limit=50)
    
    if not results['items']:
        print("[WARNING] Cronologia Spotify vuota.")
        return

    # 2. ESTRAZIONE SPOTIFY IDs
    spotify_ids = []
    tracks_data = []
    
    for item in results['items']:
        track = item['track']
        spotify_ids.append(track['id'])
        tracks_data.append({
            'id': track['id'],
            'name': track['name'],
            'artist': track['artists'][0]['name'],
            'played_at': item['played_at']
        })
    
    print(f"[INFO] Trovate {len(spotify_ids)} tracce. Inizio recupero dati da Reccobeats...")

    # 3. OTTIENI MAPPING SPOTIFY ID -> RECCOBEATS ID
    # Dividiamo in batch di 10 per non sovraccaricare l'API
    batch_size = 10
    all_mappings = {}
    
    for i in range(0, len(spotify_ids), batch_size):
        batch = spotify_ids[i:i+batch_size]
        print(f"[INFO] Processing batch {i//batch_size + 1}/{(len(spotify_ids)-1)//batch_size + 1}...")
        
        mapping = get_reccobeats_track_info(batch)
        all_mappings.update(mapping)
        
        # Piccola pausa per rispettare i rate limits
        time.sleep(0.5)
    
    print(f"[INFO] Trovate {len(all_mappings)}/{len(spotify_ids)} tracce su Reccobeats.")

    # 4. OTTIENI AUDIO FEATURES
    final_tracks = []
    features_found = 0
    
    for track_info in tracks_data:
        spotify_id = track_info['id']
        track_entry = track_info.copy()
        
        # Controlla se abbiamo l'ID di Reccobeats per questa traccia
        if spotify_id in all_mappings:
            reccobeats_id = all_mappings[spotify_id]
            
            # Ottieni le audio features
            features = get_audio_features(reccobeats_id)
            
            if features:
                # Aggiungi tutte le features al track_entry
                track_entry.update(features)
                track_entry['source'] = 'reccobeats'
                features_found += 1
            else:
                track_entry['source'] = 'not_found'
        else:
            track_entry['source'] = 'not_in_reccobeats'
        
        final_tracks.append(track_entry)
        
        # Piccola pausa tra le chiamate
        time.sleep(0.3)

    # 5. SALVATAGGIO
    print(f"[INFO] Processo completato. Audio features trovate: {features_found}/{len(final_tracks)}")
    
    df_out = pd.DataFrame(final_tracks)
    
    # Salvataggio su CSV
    output_path = os.path.join("data", "user_history.csv")
    os.makedirs("data", exist_ok=True)
    
    df_out.to_csv(output_path, index=False)
    
    print(f"[SUCCESS] File salvato correttamente: {output_path}")
    
    # Anteprima
    pd.set_option('display.max_rows', 5)
    pd.set_option('display.max_columns', 10)
    print("\nAnteprima dati:")
    print(df_out[['name', 'artist', 'source']].head(10))
    
    # Mostra alcune features se disponibili
    feature_cols = ['energy', 'valence', 'danceability', 'tempo']
    available = [c for c in feature_cols if c in df_out.columns]
    if available:
        print("\nAlcune audio features:")
        print(df_out[['name'] + available].head(5))

if __name__ == "__main__":
    fetch_history()