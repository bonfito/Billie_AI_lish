import pandas as pd
import numpy as np
from oracle import MusicOracle

def run_prediction():
    print(" Caricamento cronologia ascolti (Spotify + ReccoBeats)...")
    
    # Il Ruolo 1 deve generare questo file nel branch data&vector
    try:
        df = pd.read_csv('data/last_50_songs_features.csv')
    except FileNotFoundError:
        print(" Errore: File CSV non trovato. Chiedi al Ruolo 1 di generarlo!")
        return

    # Selezioniamo solo le colonne numeriche (le 9 feature)
    # Assicurati che l'ordine delle colonne nel CSV sia quello corretto
    feature_columns = ['energy', 'valence', 'danceability', 'acousticness', 
                       'instrumentalness', 'liveness', 'speechiness', 'loudness', 'tempo']
    
    history_data = df[feature_columns].values
    
    # Inizializza l'Oracle
    oracle = MusicOracle()
    # Qui potresti caricare i pesi se li hai salvati: 
    # oracle.load_state_dict(torch.load('model_weights.pth'))
    
    print(f"🧬 Elaborazione di {len(history_data)} brani...")
    next_dna = oracle.predict_next_dna(history_data)
    
    print("\n✨ PROSSIMA CANZONE PREDETTA (DNA):")
    for name, value in zip(feature_columns, next_dna):
        print(f"- {name:16}: {value:.4f}")
        
    print("\n🚀 Ora passa questo vettore al Ruolo 3 per la ricerca KNN!")
    # Salvataggio opzionale per il Ruolo 3
    np.save('data/predicted_dna.npy', next_dna)

if __name__ == "__main__":
    run_prediction()