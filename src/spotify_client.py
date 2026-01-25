import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
from dotenv import load_dotenv

# Carica variabili d'ambiente
load_dotenv()

CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI", "http://localhost:8501")

# Nome della Playlist
PLAYLIST_NAME = "Billie AI-lish Discovery" 
# Percorso dove salvare l'ID della playlist per non perderlo
ID_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'playlist_id.txt')

def get_spotify_client():
    if not CLIENT_ID or not CLIENT_SECRET:
        return None
    
    scope = "playlist-modify-public playlist-modify-private user-library-read"
    auth_manager = SpotifyOAuth(
        client_id=CLIENT_ID, client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI, scope=scope, cache_path=".spotify_cache"
    )
    return spotipy.Spotify(auth_manager=auth_manager)

def get_or_create_playlist_id(sp, user_id):
    """
    Logica Intelligente:
    1. Cerca se abbiamo l'ID salvato nel file playlist_id.txt
    2. Se c'è, controlla se è valido.
    3. Se non c'è, cerca su Spotify.
    4. Se non esiste su Spotify, la crea.
    """
    # 1. Controlla Cache Locale
    if os.path.exists(ID_FILE_PATH):
        with open(ID_FILE_PATH, 'r') as f:
            cached_id = f.read().strip()
        # Verifichiamo se esiste ancora su Spotify (opzionale ma sicuro)
        try:
            sp.playlist(cached_id)
            return cached_id # È valido!
        except:
            pass # ID vecchio o playlist cancellata, procediamo

    # 2. Cerca manualmente su Spotify (Fallback)
    playlists = sp.current_user_playlists(limit=50)
    for item in playlists['items']:
        if item['name'] == PLAYLIST_NAME:
            # Trovata! Salviamo l'ID per la prossima volta
            with open(ID_FILE_PATH, 'w') as f:
                f.write(item['id'])
            return item['id']

    # 3. Crea Nuova Playlist
    new_playlist = sp.user_playlist_create(
        user=user_id, 
        name=PLAYLIST_NAME, 
        public=False, 
        description="Playlist generata da Billie AI-lish"
    )
    
    # Salva ID su file
    with open(ID_FILE_PATH, 'w') as f:
        f.write(new_playlist['id'])
        
    return new_playlist['id']

def add_track_to_playlist(track_id):
    try:
        sp = get_spotify_client()
        if not sp: return False, "Chiavi API mancanti"
        
        user_id = sp.current_user()['id']
        
        # Recupera l'ID univoco (dal file o da Spotify)
        playlist_id = get_or_create_playlist_id(sp, user_id)
        
        # Aggiungi traccia
        track_uri = f"spotify:track:{track_id}"
        sp.playlist_add_items(playlist_id, [track_uri])
        
        return True, PLAYLIST_NAME
        
    except Exception as e:
        return False, str(e)

def get_track_cover(track_id):
    try:
        sp = get_spotify_client()
        if not sp: return None
        return sp.track(track_id)['album']['images'][1]['url']
    except:
        return None