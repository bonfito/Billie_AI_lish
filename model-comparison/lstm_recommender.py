import pandas as pd
import numpy as np
import torch
import os
import sys

# Import architettura LSTM
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from architectures import BillieLSTM


class LSTMRecommender:
    """
    Sistema di raccomandazione basato su LSTM.
    
    Pipeline:
    1. Carica storico ascolti (user_history.csv)
    2. Usa LSTM per predire audio features prossima canzone
    3. Cerca nel database le K canzoni più simili
    4. Filtra già ascoltate e restituisce raccomandazioni
    """
    
    def __init__(
        self,
        model_path=None,
        tracks_db_path=None,
        seq_length=20,
        device=None
    ):
        """
        Inizializza il recommender.
        """
        
        # CONFIGURAZIONE PATHS
        data_dir = os.path.join(current_dir, '..', 'data')
        models_dir = os.path.join(data_dir, 'trained_models')
        
        # Path modello LSTM
        if model_path is None:
            # Cerca modello migliore (BillieLSTM_500 o _250)
            candidates = [
                os.path.join(models_dir, 'BillieLSTM_500_best.pth'),
                os.path.join(models_dir, 'BillieLSTM_250_best.pth'),
                os.path.join(models_dir, 'BillieLSTM_50_best.pth')
            ]
            for c in candidates:
                if os.path.exists(c):
                    model_path = c
                    break
            
            if model_path is None:
                raise FileNotFoundError(
                    "Nessun modello LSTM trovato in data/trained_models/\n"
                    "Esegui prima run-experiment.py per addestrare il modello."
                )
        
        self.model_path = model_path
        
        # Path database canzoni
        if tracks_db_path is None:
            # Cerca tracks_processed.csv o tracks_db.csv
            tracks_db_path = os.path.join(data_dir, 'tracks_processed.csv')
            if not os.path.exists(tracks_db_path):
                tracks_db_path = os.path.join(data_dir, 'tracks_db.csv')
        
        self.tracks_db_path = tracks_db_path
        
        # CONFIGURAZIONE MODELLO
        self.seq_length = seq_length
        self.audio_features = [
            'energy', 'valence', 'danceability', 'tempo', 'loudness',
            'speechiness', 'acousticness', 'instrumentalness', 'liveness'
        ]
        self.n_features = len(self.audio_features)
        
        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"LSTM Recommender Inizializzato")
        print(f"Device: {self.device}")
        print(f"Modello: {os.path.basename(self.model_path)}")
        
        # CARICAMENTO MODELLO LSTM
        self.model = BillieLSTM(
            input_size=self.n_features,
            hidden_size=128,
            num_layers=2
        )
        
        # Carica pesi addestrati
        try:
            state_dict = torch.load(
                self.model_path,
                map_location=self.device,
                weights_only=True
            )
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()  # Modalità inferenza
            print(f"Modello caricato con successo")
        except Exception as e:
            raise RuntimeError(f"Errore caricamento modello: {e}")
        
        # CARICAMENTO DATABASE CANZONI
        print(f"Caricamento database canzoni...")
        print(f"Path: {self.tracks_db_path}")
        
        if not os.path.exists(self.tracks_db_path):
            raise FileNotFoundError(
                f"Database canzoni non trovato: {self.tracks_db_path}"
            )
        
        try:
            self.df_tracks = pd.read_csv(self.tracks_db_path, low_memory=False)
            
            # Normalizza nomi colonne
            self.df_tracks.columns = self.df_tracks.columns.str.lower().str.strip()
            
            # Rinomina colonne comuni
            renames = {
                'artists': 'artist',
                'track_name': 'name',
                'song': 'name',
                'genre': 'genres'
            }
            self.df_tracks.rename(columns=renames, inplace=True)
            
            # Verifica feature audio
            missing_features = [f for f in self.audio_features 
                              if f not in self.df_tracks.columns]
            
            if missing_features:
                print(f"Feature mancanti: {missing_features}")
                print(f"Imposto valori di default (0.5)")
                for feat in missing_features:
                    self.df_tracks[feat] = 0.5
            
            # Riempimento NaN
            self.df_tracks[self.audio_features] = \
                self.df_tracks[self.audio_features].fillna(0.5)
            
            # Crea matrice audio features (per similarità)
            self.audio_matrix = self.df_tracks[self.audio_features].values.astype('float32')
            
            # Normalizza per cosine similarity
            norms = np.linalg.norm(self.audio_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10  # Evita divisione per zero
            self.audio_matrix_normalized = self.audio_matrix / norms
            
            print(f"Database caricato: {len(self.df_tracks):,} canzoni")
            print(f"Features: {', '.join(self.audio_features)}")
            
        except Exception as e:
            raise RuntimeError(f"Errore caricamento database: {e}")
    
    
    # PREDIZIONE CON LSTM
    def predict_next_song(self, user_history_df):
        """
        Predice audio features della prossima canzone usando LSTM.
        """
        
        if user_history_df is None or user_history_df.empty:
            # Nessuno storico -> Predizione neutra
            print("Nessuno storico disponibile, predizione neutra")
            return np.array([0.5] * self.n_features)
        
        # Normalizza colonne
        history = user_history_df.copy()
        history.columns = history.columns.str.lower().str.strip()
        
        # Estrai ultime seq_length canzoni
        recent = history.tail(self.seq_length)
        
        # Verifica feature disponibili
        available_features = [f for f in self.audio_features if f in recent.columns]
        
        if not available_features:
            print("Nessuna audio feature nello storico, predizione neutra")
            return np.array([0.5] * self.n_features)
        
        # Crea sequenza input
        sequence = recent[available_features].fillna(0.5).values
        
        # Padding se necessario (con zeri all'inizio)
        if len(sequence) < self.seq_length:
            pad_length = self.seq_length - len(sequence)
            padding = np.zeros((pad_length, len(available_features)))
            sequence = np.vstack([padding, sequence])
        
        # Padding features se mancanti
        if len(available_features) < self.n_features:
            # Aggiungi colonne a 0.5 per feature mancanti
            missing_count = self.n_features - len(available_features)
            missing_cols = np.full((self.seq_length, missing_count), 0.5)
            sequence = np.hstack([sequence, missing_cols])
        
        # Converti a tensor PyTorch
        # Shape: (1, seq_length, n_features) per batch
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
        x = x.to(self.device)
        
        # Predizione
        with torch.no_grad():
            prediction = self.model(x)  # Shape: (1, n_features)
        
        # Converti a numpy
        predicted_features = prediction.cpu().numpy().flatten()
        
        # Clipping valori [0, 1] (sicurezza)
        predicted_features = np.clip(predicted_features, 0.0, 1.0)
        
        return predicted_features
    
    
    # RACCOMANDAZIONE
    def recommend(
        self,
        user_history_df,
        k=20,
        exclude_listened=True,
        session_blacklist=None
    ):
        """
        Genera raccomandazioni basate su predizione LSTM.
        """
        
        print("\n" + "="*70)
        print("GENERAZIONE RACCOMANDAZIONI")
        print("="*70)
        
        # STEP 1: PREDIZIONE LSTM
        print("\nSTEP 1: Predizione Audio Features con LSTM")
        predicted_features = self.predict_next_song(user_history_df)
        
        print(f"Predizione completata:")
        for i, feat in enumerate(self.audio_features):
            print(f"- {feat:15s}: {predicted_features[i]:.3f}")
        
        # STEP 2: CALCOLO SIMILARITÀ
        print(f"\nSTEP 2: Calcolo Similarità con {len(self.df_tracks):,} canzoni")
        
        # Normalizza predizione
        pred_norm = predicted_features / (np.linalg.norm(predicted_features) + 1e-10)
        
        # Cosine similarity: dot product con matrice normalizzata
        similarity_scores = np.dot(
            self.audio_matrix_normalized,
            pred_norm
        )
        
        # Aggiungi score al dataframe
        pool = self.df_tracks.copy()
        pool['similarity_score'] = similarity_scores
        
        print(f"Similarità massima: {similarity_scores.max():.4f}")
        print(f"Similarità media:   {similarity_scores.mean():.4f}")
        
        # STEP 3: FILTRI
        print(f"\nSTEP 3: Applicazione Filtri")
        
        exclude_ids = set()
        
        # Filtro: Canzoni già ascoltate
        if exclude_listened and user_history_df is not None and not user_history_df.empty:
            if 'id' in user_history_df.columns:
                listened_ids = set(user_history_df['id'].dropna().unique())
                exclude_ids.update(listened_ids)
                print(f"- Escluse {len(listened_ids)} canzoni già ascoltate")
        
        # Filtro: Blacklist sessione
        if session_blacklist:
            exclude_ids.update(session_blacklist)
            print(f"- Escluse {len(session_blacklist)} canzoni dalla blacklist")
        
        # Applica filtri
        if exclude_ids:
            pool = pool[~pool['id'].isin(exclude_ids)]
        
        print(f"Pool finale: {len(pool):,} canzoni")
        
        # STEP 4: SELEZIONE TOP K
        print(f"\nSTEP 4: Selezione Top {k}")
        
        # Ordina per similarità decrescente
        pool = pool.sort_values('similarity_score', ascending=False)
        
        # Prendi top K
        recommendations = pool.head(k).copy()
        
        # Converti score a percentuale
        recommendations['match_percentage'] = (
            recommendations['similarity_score'] * 100
        ).round(1)
        
        # Aggiungi ranking
        recommendations['rank'] = range(1, len(recommendations) + 1)
        
        # Reset index
        recommendations = recommendations.reset_index(drop=True)
        
        # STEP 5: OUTPUT
        print(f"\nRaccomandazioni generate: {len(recommendations)}")
        print("\n" + "-"*70)
        print("TOP 10 RACCOMANDAZIONI:")
        print("-"*70)
        
        display_cols = ['rank', 'name', 'artist', 'match_percentage']
        available_cols = [c for c in display_cols if c in recommendations.columns]
        
        if not recommendations.empty:
            for idx, row in recommendations.head(10).iterrows():
                rank = row.get('rank', idx+1)
                name = row.get('name', 'Unknown')
                artist = row.get('artist', 'Unknown')
                score = row.get('match_percentage', 0)
                
                print(f"{rank:2d}. {name[:40]:<40} - {artist[:20]:<20} ({score:.1f}%)")
        
        print("="*70 + "\n")
        
        return recommendations, predicted_features
    
    
    # UTILITY: ANALISI PREDIZIONE
    def analyze_prediction(self, user_history_df):
        """
        Analizza la predizione LSTM rispetto allo storico.
        """
        
        predicted = self.predict_next_song(user_history_df)
        
        if user_history_df is None or user_history_df.empty:
            return {
                'predicted': predicted,
                'history_mean': None,
                'difference': None
            }
        
        # Media features storiche
        history = user_history_df.copy()
        history.columns = history.columns.str.lower().str.strip()
        
        available = [f for f in self.audio_features if f in history.columns]
        history_mean = history[available].mean().values
        
        # Differenza
        diff = predicted[:len(available)] - history_mean
        
        return {
            'predicted': predicted,
            'history_mean': history_mean,
            'difference': diff,
            'feature_names': available
        }


# ESEMPIO USO
if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("LSTM RECOMMENDER - TEST")
    print("="*70 + "\n")
    
    # 1. INIZIALIZZAZIONE
    try:
        recommender = LSTMRecommender()
    except Exception as e:
        print(f"Errore inizializzazione: {e}")
        sys.exit(1)
    
    # 2. CARICAMENTO STORICO UTENTE
    data_dir = os.path.join(current_dir, '..', 'data')
    history_path = os.path.join(data_dir, 'user_history.csv')
    
    if os.path.exists(history_path):
        print(f"Caricamento storico da: {history_path}")
        user_history = pd.read_csv(history_path)
        print(f"Caricate {len(user_history)} canzoni storiche\n")
    else:
        print("Nessuno storico trovato, uso predizione neutra\n")
        user_history = pd.DataFrame()
    
    # 3. GENERAZIONE RACCOMANDAZIONI
    recommendations, predicted = recommender.recommend(
        user_history_df=user_history,
        k=20,
        exclude_listened=True
    )
    
    # 4. ANALISI PREDIZIONE
    if not user_history.empty:
        print("\nANALISI PREDIZIONE:")
        print("-"*70)
        
        analysis = recommender.analyze_prediction(user_history)
        
        if analysis['history_mean'] is not None:
            print(f"{'Feature':<15} {'Media Storico':<15} {'Predizione':<15} {'Differenza':<15}")
            print("-"*70)
            
            for i, feat in enumerate(analysis['feature_names']):
                hist = analysis['history_mean'][i]
                pred = analysis['predicted'][i]
                diff = analysis['difference'][i]
                
                arrow = ">" if diff > 0 else "<" if diff < 0 else "="
                
                print(f"{feat:<15} {hist:>6.3f}         {pred:>6.3f}         {arrow} {abs(diff):>5.3f}")
        
        print("="*70)
    
    print("\nTest completato!\n")