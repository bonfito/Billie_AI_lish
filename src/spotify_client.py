import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI")

PLAYLIST_NAME = "Billie AI-lish Discovery" 

def get_spotify_client():
    if not CLIENT_ID or not CLIENT_SECRET:
        return None
    
    # AGGIUNTO: playlist-read-private per essere sicuri di "vedere" la playlist creata
    scope = "playlist-modify-public playlist-modify-private user-library-read playlist-read-private"
    
    auth_manager = SpotifyOAuth(
        client_id=CLIENT_ID, 
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI, 
        scope=scope, 
        cache_path=".spotify_cache"
    )
    return spotipy.Spotify(auth_manager=auth_manager)

def get_or_create_playlist_id(sp, user_id):
    """
    Cerca la playlist in modo robusto ignorando maiuscole e spazi.
    """
    limit = 50
    offset = 0
    target_name_clean = PLAYLIST_NAME.lower().strip() #
    
    print(f"🔍 Ricerca playlist '{target_name_clean}' per l'utente {user_id}...")

    while True:
        playlists = sp.current_user_playlists(limit=limit, offset=offset)
        
        for item in playlists['items']:
            current_name_clean = item['name'].lower().strip()
            
            # Confronto tollerante
            if current_name_clean == target_name_clean:
                print(f"✅ Trovata corrispondenza: {item['name']} (ID: {item['id']})")
                return item['id']
        
        if len(playlists['items']) < limit:
            break
        offset += limit

    print(f"🆕 Playlist non trovata. Creazione di '{PLAYLIST_NAME}'...")
    new_playlist = sp.user_playlist_create(
        user=user_id, 
        name=PLAYLIST_NAME, 
        public=False, 
        description="Playlist personale generata da Billie AI-lish"
    )
    return new_playlist['id']

def add_track_to_playlist(track_id):
    try:
        sp = get_spotify_client()
        if not sp: return False, "Chiavi API mancanti"
        
        user_info = sp.current_user()
        user_id = user_info['id']
        
        playlist_id = get_or_create_playlist_id(sp, user_id)
        
        track_uri = f"spotify:track:{track_id}"
        sp.playlist_add_items(playlist_id, [track_uri])
        
        return True, PLAYLIST_NAME
        
    except Exception as e:
        print(f"❌ Errore in add_track_to_playlist: {e}")
        return False, str(e)

def get_track_cover(track_id):
    try:
        sp = get_spotify_client()
        if not sp: return None
        return sp.track(track_id)['album']['images'][1]['url']
    except:
        return None
    
def get_track_details(track_id):
    try:
        sp = get_spotify_client()
        if not sp: return "unknown", 0
        track_info = sp.track(track_id)
        if not track_info: return "unknown", 0
        popularity = track_info.get('popularity', 0)
        artist_id = track_info['artists'][0]['id']
        artist_info = sp.artist(artist_id)
        genres = artist_info.get('genres', [])
        return (genres[0], popularity) if genres else ("pop", popularity)
    except Exception as e:
        print(f" Errore get_track_details: {e}")
        return "unknown", 0