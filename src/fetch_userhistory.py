import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import requests
from dotenv import load_dotenv
import time
import numpy as np

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
        # Se l'API dà errore, non crashare ma ritorna vuoto
        if response.status_code != 200:
            return {}
            
        data = response.json()
        
        mapping = {}
        # La struttura può variare (content, data, tracks), controlliamo 'content' come nel tuo script
        items = data.get('content', [])
        
        for track in items:
            # La tua intuizione geniale: estrarre l'ID dall'href
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
    Ottiene le audio features di una traccia singola usando l'ID di Reccobeats.
    """
    url = f"https://api.reccobeats.com/v1/track/{reccobeats_id}/audio-features"
    
    try:
        response = requests.get(url, timeout=5) # Timeout breve per non bloccare tutto
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None

def fetch_history():
    # effettuo connesione a spotify
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

    # estrazione dati base
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
    
    print(f"[INFO] Trovate {len(spotify_ids)} tracce. Interrogo Recco Beats...")

    # 3. MAPPING (BATCH)
    #traduzione degli id
    batch_size = 20
    all_mappings = {}
    
    for i in range(0, len(spotify_ids), batch_size):
        batch = spotify_ids[i:i+batch_size]
        mapping = get_reccobeats_track_info(batch)
        all_mappings.update(mapping)
        time.sleep(0.2)
    
    print(f"[INFO] Tradotti {len(all_mappings)}/{len(spotify_ids)} ID.")

    # recupero le features
    final_tracks = []
    features_found = 0
    
    print("[INFO] Scaricamento features audio...")
    
    for track_info in tracks_data:
        sp_id = track_info['id']
        
        # Se abbiamo trovato l'ID Recco, chiediamo le features
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
        time.sleep(0.1) #in modo da evitare di mandare troppe richieste

    
    df = pd.DataFrame(final_tracks)
    
    # Lista delle colonne che l'AI si aspetta 
    required_cols = ['energy', 'valence', 'danceability', 'tempo', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness']
    
    #creazione delle colonne nel caso in cui le colonne non siano state trovate da recco (quando non trova il brano)
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan # Mettiamo NaN temporaneo

    # Riempimento buchi (Fallback):
    # Se una canzone non è stata trovata, mettiamo 0.5 (o la media di quelle trovate) in modo da non far crashare AI
    df[required_cols] = df[required_cols].fillna(df[required_cols].mean(numeric_only=True))
    df[required_cols] = df[required_cols].fillna(0.5) # Se tutto è NaN, metti 0.5
    
    # Salvataggio
    output_path = os.path.join("data", "user_history.csv")
    
    # Ordiniamo le colonne per averle belle pulite
    final_cols = ['id', 'name', 'artist', 'played_at', 'source'] + required_cols
    # Selezioniamo solo le colonne che esistono nel df
    cols_to_save = [c for c in final_cols if c in df.columns]
    
    df[cols_to_save].to_csv(output_path, index=False)
    
    print("-" * 30)
    print(f"[SUCCESS] Salvato in: {output_path}")
    print(f"Features reali trovate: {features_found}/{len(spotify_ids)}")
    
    # Anteprima
    pd.set_option('display.max_rows', 5)
    print("\nAnteprima:")
    print(df[['name', 'source', 'energy', 'tempo']].head())

if __name__ == "__main__":
    fetch_history()