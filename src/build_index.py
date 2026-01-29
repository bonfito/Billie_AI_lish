import pandas as pd
import numpy as np
import faiss
import sqlite3
import os

# --- CONFIGURAZIONE ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Risaliamo di un livello per andare nella cartella 'data'
DATA_DIR = os.path.normpath(os.path.join(CURRENT_DIR, '..', 'data'))

# Percorsi File
INPUT_CSV = os.path.join(DATA_DIR, 'tracks_processed.csv')
DB_PATH = os.path.join(DATA_DIR, 'tracks.db')
INDEX_PATH = os.path.join(DATA_DIR, 'tracks.index')

# Colonne Audio da vettorizzare
AUDIO_COLS = ['energy', 'valence', 'danceability', 'tempo', 'loudness', 
              'speechiness', 'acousticness', 'instrumentalness', 'liveness']

def build():
    print(f"📂 Lettura dataset da: {INPUT_CSV}")
    
    if not os.path.exists(INPUT_CSV):
        print(f"❌ ERRORE: File non trovato in {INPUT_CSV}")
        return

    # Caricamento dati
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    
    # 1. PULIZIA DATI
    # Rimuoviamo duplicati basati sull'ID di Spotify
    df = df.drop_duplicates(subset=['id']).reset_index(drop=True)
    
    # Gestione valori mancanti nelle colonne audio
    for col in AUDIO_COLS:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    print(f"📊 Totale brani processati: {len(df)}")

    # --- 2. CREAZIONE INDICE FAISS ---
    print("⚙️ Creazione Indice FAISS...")
    
    # Estraiamo la matrice numpy delle feature
    vectors = np.ascontiguousarray(df[AUDIO_COLS].values.astype('float32'))
    
    # --- PUNTO FONDAMENTALE PER COSINE SIMILARITY ---
    # Anche se i dati sono MinMax scaled (0-1), per usare Inner Product come Cosine Similarity
    # i vettori DEVONO avere norma L2 unitaria.
    faiss.normalize_L2(vectors)
    
    # Creiamo l'indice Flat Inner Product (Prodotto Scalare)
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    
    # Aggiungiamo i vettori all'indice
    index.add(vectors)
    
    # Salviamo l'indice su disco
    faiss.write_index(index, INDEX_PATH)
    print(f" Indice FAISS salvato in {INDEX_PATH}")

    # --- 3. CREAZIONE DATABASE SQLITE ---
    print(" Creazione Database SQLite...")
    
    # Rimuoviamo il vecchio DB se esiste
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        
    conn = sqlite3.connect(DB_PATH)
    
    # Aggiungiamo la colonna 'faiss_id' che corrisponde all'indice di riga (0, 1, 2...)
    # Questo serve per collegare i risultati di FAISS ai metadati
    df['faiss_id'] = df.index
    
    # Rinominiamo 'id' in 'spotify_id' per chiarezza nel database SQL
    df_sql = df.rename(columns={'id': 'spotify_id'})
    
    # Salviamo in SQLite (chunksize aiuta se il file è grande)
    df_sql.to_sql('tracks', conn, index=False, chunksize=10000)
    
    # Creiamo indici SQL per rendere istantaneo il recupero dei dati
    print(" Creazione indici SQL...")
    conn.execute('CREATE INDEX idx_faiss_id ON tracks (faiss_id)')
    conn.execute('CREATE INDEX idx_spotify_id ON tracks (spotify_id)')
    
    conn.close()
    print(f" Database SQLite salvato in {DB_PATH}")
    print("-" * 30)
    print(" OTTIMIZZAZIONE COMPLETATA CON SUCCESSO!")

if __name__ == "__main__":
    build()