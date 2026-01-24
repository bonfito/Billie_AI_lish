import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
import time
from dotenv import load_dotenv
import math

load_dotenv()

# --- AGGIUNGI QUESTO BLOCCO ---
client_id = os.getenv("SPOTIPY_CLIENT_ID")
print(f"\n🔍 DEBUG: Sto usando il Client ID che inizia con: {client_id[:5]}...")
print(f"🔍 DEBUG: Se questo ID è quello vecchio, RIAVVIA IL PC o chiudi/riapri VSCode.\n")
time.sleep(3)
# ------------------------------

# CONFIGURAZIONE DI SICUREZZA
BATCH_SIZE = 50
SAVE_INTERVAL = 1000  # Salviamo spesso
SLEEP_TIME = 0.8      # Quasi 1 secondo di pausa tra ogni chiamata (FONDAMENTALE)

def enrich_safe():
    print("🛡️ AVVIO ARRICCHIMENTO 'SAFE MODE' (Anti-Ban)...")
    
    input_path = os.path.join("data", "tracks_db.csv")
    output_path = os.path.join("data", "tracs_dbFinale.csv") # Continuiamo a usare lo stesso file

    # 1. CARICAMENTO E RESUME
    # Lo script è intelligente: riprende da dove il Turbo si è schiantato
    if os.path.exists(output_path):
        print(f"🔄 Trovato file parziale. Riprendo il lavoro...")
        df = pd.read_csv(output_path)
    else:
        print("🆕 Inizio da zero.")
        df = pd.read_csv(input_path)
        # Pulizia iniziale (se riparti da zero)
        if 'speechiness' in df.columns: df = df[df['speechiness'] < 0.66]
        if 'duration_ms' in df.columns: df = df[(df['duration_ms'] > 60000) & (df['duration_ms'] < 900000)]
        df['popularity'] = -1
        df['artist_genres'] = "pending"
        df.to_csv(output_path, index=False)

    # Indici da fare
    mask_todo = df['popularity'] == -1
    indices_to_process = df.index[mask_todo].tolist()
    total_to_do = len(indices_to_process)
    
    if total_to_do == 0:
        print("✅ Database già completo!")
        return

    print(f"⚠️ Righe rimanenti: {total_to_do}")
    print("N.B. Andrò piano per non far arrabbiare Spotify. Lasciami lavorare in background.")

    # 2. CLIENT SPOTIFY (Usa le NUOVE credenziali nel .env!)
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
        scope="user-read-private",
        requests_timeout=20
    ))

    # Cache artisti in memoria (per risparmiare chiamate)
    artist_cache = {}

    # 3. CICLO SINGOLO (NO THREAD)
    num_batches = math.ceil(total_to_do / BATCH_SIZE)
    processed_count = 0
    save_counter = 0
    start_time = time.time()

    for i in range(num_batches):
        start = i * BATCH_SIZE
        end = start + BATCH_SIZE
        current_indices = indices_to_process[start:end]
        
        # Recupero ID
        track_ids = df.loc[current_indices, 'id'].tolist()
        valid_ids = [t for t in track_ids if isinstance(t, str) and len(t) > 0]

        if not valid_ids:
            continue

        try:
            # --- CHIAMATA 1: TRACCE ---
            tracks_resp = sp.tracks(valid_ids)
            
            track_info_temp = {}
            artists_to_fetch = set()

            for track in tracks_resp['tracks']:
                if track:
                    art_id = track['artists'][0]['id'] if track['artists'] else None
                    track_info_temp[track['id']] = {'pop': track['popularity'], 'art_id': art_id}
                    
                    if art_id and art_id not in artist_cache:
                        artists_to_fetch.add(art_id)

            # --- CHIAMATA 2: ARTISTI (Solo se mancano in cache) ---
            if artists_to_fetch:
                unique_arts = list(artists_to_fetch)
                # Batch artisti
                for k in range(0, len(unique_arts), 50):
                    sub = unique_arts[k:k+50]
                    art_resp = sp.artists(sub)
                    for artist in art_resp['artists']:
                        if artist:
                            artist_cache[artist['id']] = "|".join(artist['genres']) if artist['genres'] else "unknown"
                    
                    # Mini-pausa anche qui
                    time.sleep(0.2)

            # --- AGGIORNAMENTO DATAFRAME ---
            for idx, t_id in zip(current_indices, track_ids):
                if t_id in track_info_temp:
                    info = track_info_temp[t_id]
                    df.at[idx, 'popularity'] = info['pop']
                    
                    art_id = info['art_id']
                    if art_id in artist_cache:
                        df.at[idx, 'artist_genres'] = artist_cache[art_id]
                    else:
                        df.at[idx, 'artist_genres'] = "unknown"
                else:
                    df.at[idx, 'popularity'] = 0
                    df.at[idx, 'artist_genres'] = "not_found"

            processed_count += len(valid_ids)
            save_counter += len(valid_ids)

        except Exception as e:
            print(f"⚠️ Errore temporaneo: {e}")
            print("💤 Attendo 10 secondi...")
            time.sleep(10)

        # LOGGING
        if (i+1) % 20 == 0:
            elapsed = time.time() - start_time
            rate = processed_count / elapsed
            print(f"Status: {processed_count}/{total_to_do} | Vel: {rate:.1f} songs/sec")

        # SALVATAGGIO
        if save_counter >= SAVE_INTERVAL:
            print("💾 Salvataggio sicuro...")
            df.to_csv(output_path, index=False)
            save_counter = 0
        
        # PAUSA OBBLIGATORIA ANTI-BAN
        time.sleep(SLEEP_TIME)

    df.to_csv(output_path, index=False)
    print("🎉 FINITO! (Sano e salvo)")

if __name__ == "__main__":
    enrich_safe()