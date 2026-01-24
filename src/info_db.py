import pandas as pd
import os

def analyze_database():
    file_path = os.path.join("data", "tracks_processed.csv")

    print("Analisi del Database Musicale Processato")
    print("=" * 60)

    if not os.path.exists(file_path):
        print(f"Errore: File {file_path} non trovato.")
        return

    # Caricamento (low_memory=False per velocità su file grandi)
    df = pd.read_csv(file_path, low_memory=False)

    # 1. TOTALI GENERALI
    total_songs = len(df)
    
    # Definizioni di "Dato Presente"
    # Popolarità > 0
    has_pop = df[df['popularity'] > 0]
    count_pop = len(has_pop)
    
    # Genere diverso da 'unknown' e non nullo
    has_genre = df[(df['genres'] != 'unknown') & (df['genres'].notna())]
    count_genre = len(has_genre)

    print(f"Totale Canzoni: {total_songs:,}")
    print("-" * 60)
    print(f"Canzoni con Popolarità (>0):   {count_pop:,} ({count_pop/total_songs:.1%})")
    print(f"Canzoni con Genere (noto):     {count_genre:,} ({count_genre/total_songs:.1%})")
    print("-" * 60)

    # 2. INCROCIO (CROSS-ANALYSIS)
    # Tra quelle che hanno popolarità, come siamo messi a generi?
    pop_with_genre = has_pop[has_pop['genres'] != 'unknown'].shape[0]
    pop_without_genre = len(has_pop) - pop_with_genre

    print("Analisi Incrociata (Solo su brani con Popolarità):")
    print(f"   - Hanno anche il Genere:    {pop_with_genre:,} ({pop_with_genre/len(has_pop):.1%})")
    print(f"   - NON hanno il Genere:      {pop_without_genre:,} ({pop_without_genre/len(has_pop):.1%})")
    print("-" * 60)

    # 3. STATISTICHE ARTISTI
    unique_artists = df['artist'].nunique()
    print(f"Totale Artisti Unici: {unique_artists:,}")
    print("\nTop 5 Artisti per numero di brani:")
    top_artists = df['artist'].value_counts().head(5)
    for artist, count in top_artists.items():
        print(f"   - {artist}: {count:,} brani")
    print("-" * 60)

    # 4. ANALISI TEMPORALE (DECADI)
    print("Distribuzione per Decenni:")
    print(f"{'Decade':<10} {'Totale':<10} {'Con Pop':<15} {'Con Genere':<15}")
    print("-" * 55)

    # Creazione colonna decade
    if 'year' in df.columns:
        df['decade'] = (df['year'] // 10) * 10
        
        # Raggruppamento
        decade_stats = df.groupby('decade').agg(
            total=('id', 'count'),
            with_pop=('popularity', lambda x: (x > 0).sum()),
            with_genre=('genres', lambda x: (x != 'unknown').sum())
        ).sort_index()

        for decade, row in decade_stats.iterrows():
            # Filtriamo decenni assurdi (es. anno 0 o futuro lontano)
            if 1900 <= decade <= 2030:
                print(f"{int(decade)}s      {row['total']:<10,} {row['with_pop']:<15,} {row['with_genre']:<15,}")
    else:
        print("Colonna 'year' non trovata.")
    
    print("-" * 60)

    # 5. INTEGRITÀ DATI (EXTRA)
    # Controllo ID duplicati
    duplicates = df.duplicated(subset=['id']).sum()
    
    # Controllo Audio Features "Flat" (Errori di normalizzazione)
    # Se una canzone ha energy=0 esatti potrebbe essere un errore del dataset originale
    zeros_energy = (df['energy'] == 0).sum()
    
    print("Controllo Qualità Dati:")
    print(f"   - ID Duplicati: {duplicates}")
    print(f"   - Valori 'Energy' a zero assoluto: {zeros_energy:,} (possibili errori o silenzio)")

    print("=" * 60)

if __name__ == "__main__":
    analyze_database()