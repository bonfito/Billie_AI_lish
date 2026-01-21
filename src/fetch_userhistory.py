import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import joblib
import re
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def clean_string(text):
    """
    Pulisce le stringhe per facilitare il confronto tra Spotify e Database.
    Rimuove parentesi, scritte 'remaster', 'live', ecc.
    """
    if not isinstance(text, str): return ""
    text = text.lower()
    # Rimuove contenuto tra parentesi tonde e quadre
    text = re.sub(r'\s*[\(\[].*?[\)\]]', '', text)
    # Rimuove trattini e tutto cio' che segue (spesso versioni/remaster)
    text = re.sub(r'\s-.*', '', text)
    return text.strip()

def fetch_history():
    # 1. CARICAMENTO DATABASE LOCALE
    db_path = os.path.join("data", "tracks_db.csv")
    
    if not os.path.exists(db_path):
        print(f"[ERRORE] Il file {db_path} non esiste.")
        return

    print("[INFO] Caricamento database locale in memoria...")
    
    try:
        # Carichiamo il dataset
        local_db = pd.read_csv(db_path)
        
        # Creiamo colonne di servizio per la ricerca (versioni pulite dei nomi)
        # Questo non modifica i dati originali, serve solo per il match
        local_db['match_title'] = local_db['name'].astype(str).apply(clean_string)
        # Per l'artista prendiamo solo il primo nome prima della virgola e puliamo
        local_db['match_artist'] = local_db['artists'].astype(str).apply(lambda x: clean_string(x.split(',')[0]))
        
        # Calcoliamo la media globale per i casi in cui la canzone non si trova
        features_cols = ['energy', 'valence', 'danceability', 'tempo', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness']
        # Filtriamo solo le colonne che esistono effettivamente nel CSV
        available_feats = [c for c in features_cols if c in local_db.columns]
        
        if not available_feats:
            print("[ERRORE] Il dataset non contiene le colonne audio necessarie.")
            return

        db_means = local_db[available_feats].mean(numeric_only=True)
        print(f"[INFO] Database indicizzato. Righe totali: {len(local_db)}")
        
    except Exception as e:
        print(f"[ERRORE] Lettura database fallita: {e}")
        return

    # 2. CONNESSIONE A SPOTIFY
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

    # 3. PROCESSO DI MATCHING
    final_tracks = []
    matches_found = 0
    
    print(f"[INFO] Inizio analisi di {len(results['items'])} brani...")

    for item in results['items']:
        track = item['track']
        raw_name = track['name']
        raw_artist = track['artists'][0]['name']
        
        # Dati base
        track_entry = {
            'id': track['id'],
            'name': raw_name,
            'artist': raw_artist,
            'played_at': item['played_at']
        }
        
        # Pulizia input per la ricerca
        search_title = clean_string(raw_name)
        search_artist = clean_string(raw_artist)

        # Logica di ricerca nel DataFrame
        # A. Filtra per titolo
        candidates = local_db[local_db['match_title'] == search_title]
        
        best_match = None
        
        if not candidates.empty:
            # B. Se ci sono candidati, filtra per artista
            # Verifica se la stringa artista del DB contiene l'artista cercato
            art_match = candidates[candidates['match_artist'].str.contains(search_artist, regex=False)]
            
            if not art_match.empty:
                best_match = art_match.iloc[0]
            else:
                # Se il titolo corrisponde ma l'artista no, prendiamo comunque il primo risultato
                # (Gestisce casi di feat. non elencati o variazioni minori)
                best_match = candidates.iloc[0]
        
        # Assegnazione valori
        if best_match is not None:
            # Match trovato
            for f in available_feats:
                track_entry[f] = best_match[f]
            track_entry['source'] = 'database'
            matches_found += 1
        else:
            # Match non trovato (Fallback)
            for f in available_feats:
                track_entry[f] = db_means[f]
            track_entry['source'] = 'fallback_mean'

        final_tracks.append(track_entry)

    # 4. NORMALIZZAZIONE E SALVATAGGIO
    print(f"[INFO] Matching completato. Trovati nel DB: {matches_found}/{len(final_tracks)}")
    
    df_out = pd.DataFrame(final_tracks)
    
    # Normalizzazione (Se disponibile lo scaler addestrato)
    scaler_path = os.path.join("data", "scaler.save")
    if os.path.exists(scaler_path):
        try:
            print("[INFO] Applicazione normalizzazione (scaler)...")
            scaler = joblib.load(scaler_path)
            
            # Attenzione: lo scaler si aspetta le colonne nello stesso ordine dell'addestramento
            # Qui assumiamo che available_feats copra le feature dello scaler
            if set(available_feats).issubset(df_out.columns):
                df_out[available_feats] = scaler.transform(df_out[available_feats])
        except Exception as e:
            print(f"[WARNING] Errore durante la normalizzazione: {e}")
            print("[INFO] I dati verranno salvati senza normalizzazione.")
    else:
        print("[INFO] Scaler non trovato. I dati verranno salvati grezzi.")

    # Salvataggio su CSV
    output_path = os.path.join("data", "user_history.csv")
    
    # Ordiniamo le colonne per pulizia
    cols_order = ['id', 'name', 'artist', 'played_at', 'source'] + available_feats
    final_cols = [c for c in cols_order if c in df_out.columns]
    
    df_out[final_cols].to_csv(output_path, index=False)
    
    print(f"[SUCCESS] File salvato correttamente: {output_path}")
    
    # Anteprima testuale
    pd.set_option('display.max_rows', 5)
    print("\nAnteprima dati:")
    print(df_out[['name', 'source', 'energy']].head())

if __name__ == "__main__":
    fetch_history()