import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

class SongRecommender:
    def __init__(self, dataset_path='clean_db.csv'):
        # 1. Caricamento del Dataset Reale
        try:
            self.df = pd.read_csv(dataset_path)
            
            # --- Rinomina Colonne (Compatibilità Ruolo 1 -> Ruolo 3) ---
            # Mappa: 'nome_nel_csv_ruolo1': 'nome_usato_nella_app'
            column_mapping = {
                'id': 'track_id',
                'name': 'track_name',
                'artists': 'artist_name'
                # 'valence', 'tempo', 'energy' di solito sono già corretti
            }
            self.df.rename(columns=column_mapping, inplace=True)
            
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{dataset_path}' non trovato. Assicurati di avere il CSV nella cartella src.")

        # --- DEFINIZIONE FEATURE (Secondo specifiche Ruolo 2) ---
        self.features = [
            'energy', 'valence', 'danceability', 'acousticness', 
            'instrumentalness', 'liveness', 'speechiness', 'loudness', 'tempo'
        ]
        
        # --- NORMALIZZAZIONE ---
        # Il KNN calcola distanze. Se 'tempo' è 120 e 'energy' è 0.8, il tempo domina tutto.
        # Dobbiamo scalare tutto tra 0 e 1.
        self.scaler = MinMaxScaler()
        
        # Creiamo un DataFrame di lavoro normalizzato
        self.df_normalized = self.df.copy()
        # Gestiamo eventuali valori mancanti riempiendoli con 0
        self.df_normalized[self.features] = self.df_normalized[self.features].fillna(0)
        self.df_normalized[self.features] = self.scaler.fit_transform(self.df_normalized[self.features])
        
        # Inizializza KNN sui dati normalizzati
        # 'min(15, len)' serve a non crashare se stiamo testando con poche righe
        n_neighbors = min(15, len(self.df))
        self.knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        self.knn.fit(self.df_normalized[self.features])

    def get_recommendations(self, target_vector, exclude_ids=[]):
        """
        target_vector: np.array con le 9 feature predette dall'AI.
        exclude_ids: lista di track_id da ignorare.
        """
        # 1. Normalizziamo il vettore target usando lo STESSO scaler del database
        # Creiamo un DF temporaneo per evitare warning di sklearn
        target_df = pd.DataFrame([target_vector], columns=self.features)
        
        # Importante: Se l'AI predice già valori 0-1 (come energy), lo scaler non farà danni.
        # Ma per 'tempo' e 'loudness' è fondamentale scalarli per confrontarli col DB.
        # Nota: Assumiamo che l'AI del Ruolo 2 predica nello spazio "reale" (es. BPM 120).
        # Se l'AI predice già normalizzato, bisognerebbe saltare questo passaggio, 
        # ma per sicurezza scaliamo per coerenza con il DB.
        target_scaled = self.scaler.transform(target_df)
        
        # 2. Cerchiamo i vicini
        distances, indices = self.knn.kneighbors(target_scaled)
        neighbors_indices = indices[0]
        
        # 3. Filtro ID già sentiti
        for idx in neighbors_indices:
            song = self.df.iloc[idx]
            # Controllo robusto sull'ID (gestisce anche numeri vs stringhe)
            if str(song['track_id']) not in [str(x) for x in exclude_ids]:
                return song 
        
        # Fallback: restituisce il primo anche se già sentito (meglio che crashare)
        return self.df.iloc[neighbors_indices[0]]