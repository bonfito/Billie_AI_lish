import pandas as pd
import os
import joblib
from sklearn.preprocessing import MinMaxScaler

def process_data():
    input_path = os.path.join("data", "tracks_db.csv")
    output_path = os.path.join("data", "tracks_processed.csv")

    scaler_path = os.path.join("data", "scaler.save")

    print("caricamento dati..")

    #lettura csv
    if os.path.exists(input_path):
        df = pd.read_csv(input_path)
    else:
        print("file track_db.csv non trovato")
        return 
    
    #seleziono audio features
    features = ['energy', 'valence', 'danceability', 'tempo', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness'] 

    #salviamo ID della canzone
    colonne_utili = ['id'] + features

    #dropna rimuove righe con dati mancanti
    df_clean = df[colonne_utili].dropna()

    print("Normalizzazione in corso..")

    scaler = MinMaxScaler()

    df_clean[features] = scaler.fit_transform(df_clean[features])

    print("salvataggio file..")
    df_clean.to_csv(output_path, index=False)

    #salviamo anche lo scaler
    joblib.dump(scaler, scaler_path)

    print("file salvato, terminato")

if __name__ == "__main__":
    process_data()