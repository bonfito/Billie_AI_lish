import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
from config import *

def get_session_data(playlist_id):
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, 
                                               redirect_uri=REDIRECT_URI, scope="playlist-read-private"))
    
    results = sp.playlist_tracks(playlist_id)
    tracks = results['items']
    
    data = []
    for item in tracks:
        t = item['track']
        f = sp.audio_features(t['id'])[0] # Ottiene le 9 feature [cite: 8, 68]
        data.append({
            'name': t['name'],
            'popularity': t['popularity'] / 100, # Normalizzazione [cite: 10, 69]
            'energy': f['energy'],
            'danceability': f['danceability'],
            'valence': f['valence'],
            'acousticness': f['acousticness'],
            'instrumentalness': f['instrumentalness'],
            'liveness': f['liveness'],
            'speechiness': f['speechiness'],
            'loudness': (f['loudness'] + 60) / 60 # Normalizzazione dB [cite: 18, 69]
        })
    
    df = pd.DataFrame(data)
    df.to_csv('training_session.csv', index=False)
    print("Dataset creato: training_session.csv")

# Usa l'ID di una tua playlist per testare
# get_session_data('ID_PLAYLIST')