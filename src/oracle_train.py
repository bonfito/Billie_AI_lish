import pandas as pd
import os 
import joblib
import numpy as np

# Import delle altre classi della repo
from src.oracle import MusicOracle
# Importiamo la logica di contesto (valanga)
try:
    from src.utils import calculate_avalanche_context
except ImportError:
    from utils import calculate_avalanche_context

def train_loop():
    print("Avvio procedura di addestramento Oracle...")
    
    # Definisco i percorsi
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')
    
    history_path = os.path.join(data_dir, "user_history.csv")
    oracle_path = os.path.join(data_dir, "oracle.pkl")

    # Carico i dati
    if not os.path.exists(history_path):
        print(f"Errore: file {history_path} non trovato. Eseguire prima fetch_userhistory.")
        return
    
    df = pd.read_csv(history_path)

    # --- MODIFICA CRITICA PER NUOVO FORMATO DATI ---
    # Il file ora contiene sia cronologia reale (con data) sia Top Tracks (potenzialmente senza data recente).
    # L'Oracle deve imparare le transizioni, quindi ci serve solo ciò che è sequenziale.
    
    if 'played_at' in df.columns:
        # Rimuoviamo righe che non hanno una data valida (Top Tracks puri non ascoltati di recente)
        df_training = df.dropna(subset=['played_at']).copy()
        
        # Convertiamo e ordiniamo cronologicamente
        df_training['played_at'] = pd.to_datetime(df_training['played_at'])
        df_training = df_training.sort_values(by='played_at', ascending=True).reset_index(drop=True)
        
        print(f"Dataset filtrato per training sequenziale: {len(df_training)} brani (esclusi Top Tracks statici).")
    else:
        print("Errore critico: colonna played_at mancante.")
        return
    
    if len(df_training) < 2:
        print("Dati insufficienti per il training (meno di 2 brani sequenziali).")
        return

    # Colonne delle audio features su cui addestrare
    feature_cols = ['energy', 'valence', 'danceability', 'tempo',
                    'loudness', 'speechiness', 'acousticness', 'instrumentalness',
                    'liveness']
    
    # Inizializzazione o Caricamento Oracle esistente
    if os.path.exists(oracle_path):
        try:
            oracle = joblib.load(oracle_path)
            print("Modello esistente caricato. Addestramento incrementale.")
        except:
            oracle = MusicOracle()
            print("Nuovo modello inizializzato.")
    else:
        oracle = MusicOracle()
        print("Nuovo modello inizializzato.")

    print("Inizio fase di apprendimento sulle transizioni...")

    # Usiamo la prima canzone per inizializzare il contesto
    first_track_features = df_training.loc[0, feature_cols].values
    current_context = first_track_features

    errors = [] 

    # Iteriamo dalla seconda canzone fino all'ultima
    for i in range(1, len(df_training)):
        # La canzone che l'utente ha effettivamente scelto dopo (target reale)
        target_track = df_training.loc[i, feature_cols].values

        # Addestramento: Dato il contesto attuale -> Prevedi il target reale
        oracle.train_incremental(current_context, target_track)

        # Calcolo errore (opzionale, per debug)
        # prediction = oracle.predict_target(current_context)
        # error = np.mean((prediction - target_track) ** 2)
        # errors.append(error)

        # Aggiornamento contesto per il passo successivo (Effetto Valanga)
        current_context = calculate_avalanche_context(current_context, target_track, n=5)

    # Salvataggio
    joblib.dump(oracle, oracle_path)
    print("Addestramento completato e modello salvato.")
    print("-" * 20)

if __name__ == "__main__":
    train_loop()