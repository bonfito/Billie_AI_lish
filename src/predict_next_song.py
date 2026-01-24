import pandas as pd
import numpy as np
from oracle import MusicOracle

def run_prediction():
    print("Analisi cronologia (Audio + Artisti)...")
    
    try:
        # Carichiamo il file processato (ora ci aspettiamo solo l'artista come metadato)
        df = pd.read_csv('data/user_history_processed.csv')
    except FileNotFoundError:
        print("Errore: Esegui fetch_history.py prima!")
        return

    # Le 9 feature audio standard
    feature_columns = ['energy', 'valence', 'danceability', 'tempo', 'loudness', 
                       'speechiness', 'acousticness', 'instrumentalness', 'liveness']
    
    # Verifichiamo che le colonne esistano nel DF prima di procedere
    available_cols = [c for c in feature_columns if c in df.columns]
    history_data = df[available_cols].values
    
    # Analisi metadati (Solo Artist, rimosso Genre)
    top_artist = "Unknown"
    if 'artist' in df.columns:
        top_artist = df['artist'].mode()[0]
    
    oracle = MusicOracle()
    
    print(f"🧬 Elaborazione DNA basata su {len(history_data)} brani...")
    print(f"⭐ Artista preferito recente: {top_artist}")
    
    # Predizione del prossimo DNA musicale tramite l'Oracle
    # Assicurati che il metodo si chiami 'predict_next_dna' o 'predict_target' nel tuo oracle.py
    try:
        next_dna = oracle.predict_next_dna(history_data)
    except AttributeError:
        # Fallback se il metodo si chiama diversamente
        next_dna = oracle.predict_target(history_data[-1] if len(history_data) > 0 else np.array([0.5]*9))
    
    print("\n✨ TARGET DNA PREDETTO:")
    for name, value in zip(available_cols, next_dna):
        print(f"- {name:16}: {value:.4f}")
        
    # Salviamo il DNA e l'artista suggerito per il Recommender (Ruolo 3)
    prediction_package = {
        'dna': next_dna,
        'suggested_artist': top_artist
    }
    
    # Salvataggio in formato .npy per essere letto facilmente dagli altri moduli
    np.save('data/prediction_package.npy', prediction_package)
    print("\n🚀 Pacchetto predizione salvato (DNA + Artista).")

if __name__ == "__main__":
    run_prediction()