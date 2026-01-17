import pandas as pd
import os
import kagglehub

def prepare_data():
    # 1. Scarica (o recupera se già scaricato) il percorso del dataset
    print("Recupero percorso dataset da kagglehub...")
    path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")
    
    # 2. Costruisce il percorso corretto del file CSV
    # Il dataset di Kaggle si chiama solitamente 'dataset.csv'
    csv_path = os.path.join(path, "dataset.csv")
    
    if not os.path.exists(csv_path):
        print(f"Errore: Il file {csv_path} non esiste.")
        return

    # 3. Caricamento del dataset [cite: 53]
    full_df = pd.read_csv(csv_path)

    # 4. Selezione delle 9 feature richieste dalle specifiche [cite: 8, 10-18]
    features = ['popularity', 'energy', 'danceability', 'valence', 'acousticness', 
                'instrumentalness', 'liveness', 'speechiness', 'loudness']

    # Prendiamo 100 brani per simulare una sessione [cite: 67]
    session_data = full_df[['track_name'] + features].head(100).copy()

    # 5. Normalizzazione (Step 1.4 delle specifiche) 
    # Popularity da 0-100 a 0-1 [cite: 10]
    session_data['popularity'] = session_data['popularity'] / 100
    
    # Normalizzazione Loudness (dB) in range 0-1 [cite: 18]
    min_l = session_data['loudness'].min()
    max_l = session_data['loudness'].max()
    session_data['loudness'] = (session_data['loudness'] - min_l) / (max_l - min_l)

    # 6. Salvataggio in CSV ordinato cronologicamente (simulato) [cite: 70]
    session_data.to_csv('training_session.csv', index=False)
    print("Successo! File 'training_session.csv' creato nella cartella del progetto.")

if __name__ == "__main__":
    prepare_data()