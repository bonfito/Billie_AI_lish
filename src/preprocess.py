import pandas as pd
import os
import joblib
import numpy as np

def process_data():
    input_path = os.path.join("data", "tracks_db.csv")
    output_path = os.path.join("data", "tracks_processed.csv")
    scaler_path = os.path.join("data", "scaler.save")

    print("Avvio elaborazione del database musicale...")
    print("-" * 60)

    # 1. CARICAMENTO
    if os.path.exists(input_path):
        print("Caricamento dataset in corso...")
        df = pd.read_csv(input_path, low_memory=False)
    else:
        print(f"Errore: File {input_path} non trovato")
        return 
    
    if not os.path.exists(scaler_path):
        print("Errore: Scaler non trovato. Esegui prima lo script della cronologia.")
        return

    # 2. ALLINEAMENTO E PULIZIA
    print("Pulizia nomi artisti e allineamento colonne...")
    
    # Rinomina colonne
    df = df.rename(columns={'artists': 'artist', 'genre': 'genres'})
    
    # Pulizia stringhe artisti: rimuove parentesi quadre e virgolette ["Artist"] -> Artist
    df['artist'] = df['artist'].astype(str).str.replace(r"['\[\]]", "", regex=True).str.strip()
    
    # Rimozione righe corrotte
    df = df.dropna(subset=['id', 'name', 'artist'])
    
    # Conversione numerica sicura della popolarità
    df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(0).astype(float)

    # =========================================================================
    # FASE DI RECUPERO DATI (Imputazione basata su Artista)
    # =========================================================================
    print("Analisi artisti per recupero dati mancanti (Popolarità e Genere)...")

    # A. Calcolo statistiche per artista
    valid_pop = df[df['popularity'] > 0]
    artist_pop_mean = valid_pop.groupby('artist')['popularity'].mean()
    
    # Identificazione artisti con dati misti (alcuni presenti, alcuni mancanti)
    df['has_pop'] = df['popularity'] > 0
    stats = df.groupby('artist')['has_pop'].agg(['count', 'sum'])
    stats.columns = ['total_songs', 'valid_songs']
    stats['missing_songs'] = stats['total_songs'] - stats['valid_songs']
    
    # Unione con la media calcolata
    stats = stats.join(artist_pop_mean.rename("avg_popularity"))
    
    # Filtro artisti recuperabili
    fixable_artists = stats[(stats['valid_songs'] > 0) & (stats['missing_songs'] > 0)]
    print(f"Artisti analizzati con dati parziali: {len(fixable_artists):,}")

    # B. CORREZIONE POPOLARITÀ
    # Creazione mappa {Artista: Media Popolarità}
    pop_map = fixable_artists['avg_popularity'].to_dict()
    
    # Selezione righe da correggere (Pop=0 E Artista noto)
    mask_fix_pop = (df['popularity'] == 0) & (df['artist'].isin(pop_map.keys()))
    
    # Applicazione correzione
    df.loc[mask_fix_pop, 'popularity'] = df.loc[mask_fix_pop, 'artist'].map(pop_map)

    # C. CORREZIONE GENERI
    # Cerchiamo il genere più frequente per artista (escludendo unknown)
    valid_genres = df[(df['genres'] != 'unknown') & (df['genres'].notna())]
    
    fixed_genres_count = 0
    if not valid_genres.empty:
        # Calcolo moda (genere più frequente)
        artist_genre_mode = valid_genres.groupby('artist')['genres'].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown"
        )
        genre_map = artist_genre_mode.to_dict()
        
        # Identificazione righe da correggere
        mask_fix_genre = (df['genres'] == 'unknown') | (df['genres'].isna())
        mask_fix_genre = mask_fix_genre & (df['artist'].isin(genre_map.keys()))
        
        # Applicazione correzione
        df.loc[mask_fix_genre, 'genres'] = df.loc[mask_fix_genre, 'artist'].map(genre_map)
        fixed_genres_count = mask_fix_genre.sum()

    print("-" * 60)
    print(f"Canzoni con popolarità recuperata (da 0 a media artista): {mask_fix_pop.sum():,}")
    print(f"Canzoni con genere recuperato (da unknown a noto): {fixed_genres_count:,}")
    print("-" * 60)

    # =========================================================================
    # 3. NORMALIZZAZIONE E SALVATAGGIO
    # =========================================================================
    audio_features = ['energy', 'valence', 'danceability', 'tempo', 'loudness', 
                      'speechiness', 'acousticness', 'instrumentalness', 'liveness']
    metadata = ['id', 'name', 'artist', 'popularity', 'year', 'genres']
    
    # Selezione finale colonne
    cols_to_keep = metadata + audio_features
    df_clean = df[[c for c in cols_to_keep if c in df.columns]].copy()
    
    # Gestione valori audio mancanti (fallback su media)
    df_clean[audio_features] = df_clean[audio_features].fillna(df_clean[audio_features].mean(numeric_only=True))
    
    # Ordinamento temporale
    if 'year' in df_clean.columns:
        df_clean = df_clean.sort_values(by='year', ascending=False)

    print("Normalizzazione feature audio con Scaler esistente...")
    try:
        scaler = joblib.load(scaler_path)
        df_clean[audio_features] = scaler.transform(df_clean[audio_features])
    except Exception as e:
        print(f"Errore critico durante la normalizzazione: {e}")
        return

    print(f"Salvataggio file processato in: {output_path}")
    df_clean.to_csv(output_path, index=False)
    print("Operazione completata.")

if __name__ == "__main__":
    process_data()