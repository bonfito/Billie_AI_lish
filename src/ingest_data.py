import pandas as pd
import numpy as np
import faiss
import sqlite3
import os 
from tqdm import tqdm

#configurazione
DATA_DIR = "data"

INPUT_CSV = os.path.join(DATA_DIR, "tracks_processed.csv")
DB_FILE = os.path.join(DATA_DIR, "tracks.db")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "tracks.index")

FEATURE_COLS = [
    'energy', 'valence', 'danceability', 'tempo', 'loudness', 
    'speechiness', 'acousticness', 'instrumentalness', 'liveness'
]

def ingest_data():
    print(f"Inizio ingestione da: {INPUT_CSV}")

    #preparazione SQlite per i metadati
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE) 

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE tracks (
            faiss_id INTEGER PRIMARY KEY,
            spotify_id TEXT,
            name TEXT,
            artist TEXT,
            popularity REAL,
            year INTEGER,
            genres TEXT
        )
    """)

    cursor.execute("CREATE INDEX idx_spotify_id ON tracks(spotify_id)")
    cursor.execute("CREATE INDEX idx_artist ON tracks(artist)")

    #preparazione FAISS, vettori
    dimension = len(FEATURE_COLS)
    index = faiss.IndexFlatL2(dimension)

    #elaborazione a blocchi
    chunk_size = 50000
    current_faiss_id = 0

    try:
        total_rows = sum(1 for _ in open(INPUT_CSV, encoding='utf-8')) - 1
    except Exception:
        total_rows = None 

    print(f"Inizio elaborazione di {total_rows} brani..")

    chunks = pd.read_csv(INPUT_CSV, chunksize=chunk_size)
    
    if total_rows:
        chunks = tqdm(chunks, total=total_rows//chunk_size+1, unit="chunk")

    for chunk in chunks:
        # Gestione vettori in faiss
        features = np.ascontiguousarray(
            chunk[FEATURE_COLS].fillna(0).astype('float32').to_numpy()
        )

        index.add(features)

        # Gestione metadati in sqlite
        #  .copy() evita il SettingWithCopyWarning di Pandas
        meta_data = chunk[['id', 'name', 'artist', 'popularity', 'year', 'genres']].copy()
        meta_data['faiss_id'] = range(current_faiss_id, current_faiss_id + len(chunk))

        # Riordiniamo le colonne per matchare la tabella SQL (faiss_id per primo)
        # Nota: nel CSV la colonna si chiama 'id', nel DB la mappiamo in 'spotify_id'
        records = meta_data[['faiss_id', 'id', 'name', 'artist', 'popularity', 'year', 'genres']].to_records(index=False).tolist()

        cursor.executemany("INSERT INTO tracks VALUES (?,?,?,?,?,?,?)", records)

        current_faiss_id += len(chunk)
    
    #salvataggio
    print("salvo databse SQLite")
    conn.commit()
    conn.close()

    print("Salvataggio indice FAISS")
    faiss.write_index(index, FAISS_INDEX_FILE)

    print("-"*20)
    print(f"Totale brani indicizzati: {current_faiss_id}")

if __name__ ==  "__main__":
    try:
        ingest_data() # <--- FIX: Aggiunte parentesi per eseguire la funzione
    except Exception as e:
        print(f"Errore durante ingestione: {e}")