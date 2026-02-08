
import os
import requests
import base64
from dotenv import load_dotenv

# Determina la directory corrente del file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Costruisci il percorso verso la root del progetto (risalendo di un livello)
# Assumendo che spotify_helper.py sia in model-comparison/ o simile, e .env sia fuori
project_root = os.path.dirname(current_dir) 
dotenv_path = os.path.join(project_root, '.env')

# Se non lo trova lì, prova nella directory corrente (caso in cui script è spostato)
if not os.path.exists(dotenv_path):
    dotenv_path = os.path.join(current_dir, '.env')

# Carica variabili ambiente specificando il path esatto
load_dotenv(dotenv_path)

SPOTIFY_CLIENT_ID = os.getenv('SPOTIPY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIPY_CLIENT_SECRET')

# Token cache (per non richiedere ad ogni chiamata)
_access_token = None


def get_spotify_token():
    """
    Ottiene access token da Spotify API.
    
    Usa Client Credentials Flow (no user login richiesto).
    
    Returns:
        str: Access token o None se errore
    """
    global _access_token
    
    # Verifica credenziali
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        return None
    
    # Se token già in cache, riusalo (dura 1 ora)
    # In produzione, implementa refresh quando scade
    if _access_token:
        return _access_token
    
    # Richiedi nuovo token
    try:
        # Endpoint token
        url = "https://accounts.spotify.com/api/token"
        
        # Header con credenziali base64
        auth_str = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}"
        auth_bytes = auth_str.encode('utf-8')
        auth_base64 = base64.b64encode(auth_bytes).decode('utf-8')
        
        headers = {
            'Authorization': f'Basic {auth_base64}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        # Body
        data = {
            'grant_type': 'client_credentials'
        }
        
        # Richiesta
        response = requests.post(url, headers=headers, data=data, timeout=5)
        
        if response.status_code == 200:
            token_data = response.json()
            _access_token = token_data.get('access_token')
            return _access_token
        else:
            print(f"Errore Spotify token: {response.status_code}")
            return None
    
    except Exception as e:
        print(f"Errore connessione Spotify: {e}")
        return None


def get_track_info(track_id):
    """
    Ottiene informazioni traccia da Spotify API.
    
    Include: nome, artista, album, copertina, preview URL.
    
    Args:
        track_id (str): Spotify track ID
    
    Returns:
        dict: Informazioni traccia o None se errore
    """
    
    if not track_id:
        return None
    
    # Ottieni token
    token = get_spotify_token()
    if not token:
        return None
    
    try:
        # Endpoint traccia
        url = f"https://api.spotify.com/v1/tracks/{track_id}"
        
        headers = {
            'Authorization': f'Bearer {token}'
        }
        
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            # Estrai informazioni
            info = {
                'id': track_id,
                'name': data.get('name'),
                'artist': ', '.join([a['name'] for a in data.get('artists', [])]),
                'album': data['album'].get('name'),
                'album_cover': None,  # Popolato sotto
                'preview_url': data.get('preview_url'),  # 30s preview MP3
                'popularity': data.get('popularity'),
                'duration_ms': data.get('duration_ms'),
                'release_date': data['album'].get('release_date')
            }
            
            # Copertina album (più grande disponibile)
            images = data['album'].get('images', [])
            if images:
                # Spotify fornisce 3 sizes: 640x640, 300x300, 64x64
                # Prendiamo la più grande (primo elemento)
                info['album_cover'] = images[0]['url']
            
            return info
        
        else:
            print(f"Errore Spotify API: {response.status_code}")
            return None
    
    except Exception as e:
        print(f"Errore get_track_info: {e}")
        return None


def get_album_cover(track_id, size='medium'):
    """
    Ottiene solo URL copertina album (funzione semplificata).
    
    Args:
        track_id (str): Spotify track ID
        size (str): 'large' (640px), 'medium' (300px), 'small' (64px)
    
    Returns:
        str: URL copertina o None
    """
    
    info = get_track_info(track_id)
    
    if not info:
        return None
    
    cover_url = info.get('album_cover')
    
    if not cover_url:
        return None
    
    # Spotify fornisce sempre 640px di default
    # Per altre dimensioni, modifica URL
    if size == 'medium':
        cover_url = cover_url.replace('640x640', '300x300')
    elif size == 'small':
        cover_url = cover_url.replace('640x640', '64x64')
    
    return cover_url


# ═══════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("SPOTIFY HELPER - TEST")
    print("="*60 + "\n")
    
    # Test track: "Shape of You" by Ed Sheeran
    test_track_id = "7qiZfU4dY1lWllzX7mPBI"
    
    print(f"Testing track ID: {test_track_id}\n")
    
    # Test 1: Get token
    print("1. Getting Spotify token...")
    token = get_spotify_token()
    
    if token:
        print(f"   ✅ Token obtained: {token[:20]}...\n")
    else:
        print("   ❌ Token failed (check credentials in .env)\n")
        print("   Setup:")
        print("   1. Go to https://developer.spotify.com/dashboard")
        print("   2. Create app")
        print("   3. Copy Client ID and Secret")
        print("   4. Create .env file:")
        print("      SPOTIFY_CLIENT_ID=your_id")
        print("      SPOTIFY_CLIENT_SECRET=your_secret\n")
        exit(1)
    
    # Test 2: Get track info
    print("2. Getting track info...")
    info = get_track_info(test_track_id)
    
    if info:
        print(f"   ✅ Track: {info['name']}")
        print(f"   ✅ Artist: {info['artist']}")
        print(f"   ✅ Album: {info['album']}")
        print(f"   ✅ Cover: {info['album_cover'][:50]}...")
        print(f"   ✅ Preview: {info['preview_url'][:50] if info['preview_url'] else 'N/A'}...\n")
    else:
        print("   ❌ Failed to get track info\n")
    
    # Test 3: Get cover only
    print("3. Getting album cover (medium size)...")
    cover = get_album_cover(test_track_id, size='medium')
    
    if cover:
        print(f"   ✅ Cover URL: {cover}\n")
    else:
        print("   ❌ Failed to get cover\n")
    
    print("="*60)
    print("✅ All tests completed!")
    print("="*60 + "\n")