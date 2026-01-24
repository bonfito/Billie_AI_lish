import pandas as pd
import os
import joblib
from sklearn.preprocessing import MinMaxScaler

def process_data():
    input_path = os.path.join("data", "tracks_db.csv")
    output_path = os.path.join("data", "tracks_processed.csv")
    scaler_path = os.path.join("data", "scaler.save")

    print("Avvio elaborazione dati...")

    # Lettura csv
    if os.path.exists(input_path):
        # low_memory=False aiuta a gestire file di grandi dimensioni
        df = pd.read_csv(input_path, low_memory=False)
    else:
        print("Errore: file tracks_db.csv non trovato")
        return 
    
    # Definizione colonne
    # Features numeriche da normalizzare
    audio_features = ['energy', 'valence', 'danceability', 'tempo', 'loudness', 
                      'speechiness', 'acousticness', 'instrumentalness', 'liveness']
    
    # Metadati da conservare per il motore di raccomandazione
    metadata = ['id', 'name', 'artists', 'popularity', 'year', 'genre']
    
    # Unione delle colonne necessarie
    # Verifichiamo che esistano tutte nel df per evitare errori
    existing_cols = [c for c in metadata + audio_features if c in df.columns]
    
    # Selezione e rimozione righe con valori nulli nelle colonne chiave
    df_clean = df[existing_cols].dropna()

    print("Ordinamento dati per anno...")
    if 'year' in df_clean.columns:
        # Ordiniamo dal più recente al più vecchio (utile per visualizzare prima i brani moderni)
        df_clean = df_clean.sort_values(by='year', ascending=False)

    print("Normalizzazione feature audio...")
    scaler = MinMaxScaler()

    # Normalizziamo solo le colonne numeriche delle feature
    # I metadati (anno, popolarità, ecc.) restano invariati
    df_clean[audio_features] = scaler.fit_transform(df_clean[audio_features])

    print("Salvataggio file processato...")
    df_clean.to_csv(output_path, index=False)

    # Salviamo lo scaler per usarlo sui dati utente in futuro
    joblib.dump(scaler, scaler_path)

    print("Elaborazione terminata con successo.")

if __name__ == "__main__":
    process_data()