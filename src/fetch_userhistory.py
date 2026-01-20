import os 
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from dotenv import load_dotenv

# Carica le credenziali dal file env
load_dotenv()

def fetch_history():
    scope = "user-read-recently-played"

    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        scope=scope,
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI")
    ))

    print("Connesso a Spotify.")
    print("Scarico le ultime 50 canzoni ascoltate...")

    results = sp.current_user_recently_played(limit=50)

    if not results['items']:
        print("Nessuna canzone trovata nella cronologia recente.")
        return
    
    tracks_data = []

    for item in results['items']:
        track = item['track']
        
        tracks_data.append({
            'id': track['id'],
            'name': track['name'],
            'artist': track['artists'][0]['name'],
            'played_at': item['played_at']
        })

    df = pd.DataFrame(tracks_data)

    output_path = os.path.join("data", "user_history.csv")
    df.to_csv(output_path, index=False)

    print(f"Trovate {len(df)} canzoni.")
    print(f"Salvate in: {output_path}")
    
    # Imposta pandas per mostrare tutte le righe senza tagliarle
    pd.set_option('display.max_rows', None)
    
    print("\nElenco completo delle canzoni recuperate:")
    # Stampa l'intero DataFrame selezionando solo nome e artista
    print(df[['name', 'artist']])

if __name__ == "__main__":
    fetch_history()