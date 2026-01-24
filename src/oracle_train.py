import pandas as pd
import os 
import joblib
import numpy as np

#import delle altre classi della repo
from src.oracle import MusicOracle
from src.utils import calculate_avalanche_context

def train_loop():
    print("Avvio addestramento....")
    print("-" * 20)

    #definisco i percorsi
    history_path = os.path.join("data", "user_history.csv")
    oracle_path = os.path.join("data", "oracle.pkl")

    #carico i dati
    if not os.path.exists(history_path):
        print(f"Errore: file {history_path} non trovato, eseguire prima il fetc history")
        return
    
    df = pd.read_csv(history_path)

    #ordinamento cronologico, in modo che ai possa imparare la sequenza temporale
    if 'played_at' in df.columns:
        df['played_at'] = pd.to_datetime(df['played_at'])
        df = df.sort_values(by='played_at', ascending=True).reset_index(drop=True)
        print(f"Cronologia ordinata: {len(df)} brani caricati")
    else:
        print("Attenzione: colonna played_at mancante. Ordine potrebbe non essere corretto")
    
    #colonne delle audio features
    feature_cols = ['energy', 'valence', 'danceability', 'tempo',
                    'loudness', 'speechiness', 'acousticness', 'instrumentalness',
                    'liveness']
    
    #inizializzazione oracle
    oracle = MusicOracle()
    print("Modello di MusicOracle inizializzato")

    #loop di addestramento
    print("inizio apprendimento sequenziale")

    #usiamo la prima canzone per inizializzare il contesto
    # (non possiamo prevedere la prima canzone perché prima non c'è nulla)
    first_track_features = df.loc[0, feature_cols].values
    current_context = first_track_features

    errors = [] #per tracciare se sta imparando

    #iteriamo dalla seconda canzone fino all'ultima
    for i in range(1, len(df)):
        #la canzone che l'utente ha effettivamente scelto (target reale)
        target_track = df.loc[i, feature_cols].values

        #Addestramento
        #visto il contesto, l'utente ha scelto target track, da qui impara
        oracle.train_incremental(current_context, target_track)

        #calcolo dell'errore prima dell'update per vedere se sta effettivamente imparando
        prediction = oracle.predict_target(current_context)
        error = np.mean((prediction-target_track) ** 2)
        errors.append(error)

        #aggiornamento contesto
        #il nuovo contesto è un mix del vecchio + la nuova canzone
        current_context = calculate_avalanche_context(current_context, target_track, n=5)

    #salvataggio
    print("-"*10)
    print("Addestramento completato")

    joblib.dump(oracle, oracle_path)

if __name__ == "__main__":
    train_loop()