import pandas as pd
import os

csv_path = os.path.join("data", "tracks_db.csv")


if os.path.exists(csv_path):
    print("file trovato")

    try:
        df = pd.read_csv(csv_path, nrows=5)
        print("\n Anteprima dati: ")
        print(df[['tempo', 'loudness', 'energy']])

    except Exception as e:
        print("errore nella lettura")

else:
    print("file non trovato")

