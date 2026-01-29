import pandas as pd
import numpy as np
import os
import joblib

# --- IMPORT INTELLIGENTE ---
try:
    from utils import calculate_avalanche_context
except ImportError:
    from src.utils import calculate_avalanche_context

class SongRecommender:
    def __init__(self, dataset_path=None):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 1. Gestione Percorso Database
        if dataset_path:
            self.tracks_path = dataset_path
        else:
            # Tenta diversi percorsi standard
            path_attempt_1 = os.path.normpath(os.path.join(current_dir, '..', 'data', 'tracks_processed.csv'))
            path_attempt_2 = os.path.join(current_dir, 'tracks_processed.csv')
            
            if os.path.exists(path_attempt_1):
                self.tracks_path = path_attempt_1
            elif os.path.exists(path_attempt_2):
                self.tracks_path = path_attempt_2
            else:
                self.tracks_path = os.path.normpath(os.path.join(current_dir, '..', 'data', 'tracks_db.csv'))

        # 2. Gestione Percorso Oracle e Blacklist
        self.oracle_path = os.path.join(current_dir, '..', 'data', 'oracle.pkl')
        self.blacklist_path = os.path.join(current_dir, '..', 'data', 'blacklist.txt')

        self.audio_cols = ['energy', 'valence', 'danceability', 'tempo', 'loudness', 
                           'speechiness', 'acousticness', 'instrumentalness', 'liveness']

        # 3. Caricamento Dati
        print(f"✅ Recommender: Caricamento DB da {self.tracks_path}")
        if os.path.exists(self.tracks_path):
            # Ottimizzazione memoria: float32
            self.df_tracks = pd.read_csv(self.tracks_path, low_memory=False)
            
            # --- PRE-CALCOLO MATRICE NUMPY (Il segreto della velocità) ---
            print("⚙️ Ottimizzazione Matrice Audio...")
            # 1. Estraiamo solo i numeri e convertiamo in float32 (metà RAM)
            self.matrix = self.df_tracks[self.audio_cols].fillna(0.5).values.astype('float32')
            
            # 2. Normalizziamo subito i vettori (L2 Norm)
            # In questo modo similarity = dot_product, molto più veloce
            norm = np.linalg.norm(self.matrix, axis=1)[:, np.newaxis]
            norm[norm == 0] = 1e-10 # Evita divisione per zero
            self.matrix_normalized = self.matrix / norm
            print("✅ Motore Audio Pronto.")
            
        else:
            print(f"⚠️ ATTENZIONE: Database non trovato in {self.tracks_path}")
            self.df_tracks = pd.DataFrame()
            self.matrix_normalized = None

        # Caricamento Modello AI Oracle
        if os.path.exists(self.oracle_path):
            try:
                self.oracle = joblib.load(self.oracle_path)
            except:
                self.oracle = None
        else:
            self.oracle = None

    def _get_current_context(self, user_history_df):
        """Calcola il vettore medio pesato (Avalanche) della storia utente."""
        if user_history_df.empty:
            return np.array([0.5]*9)
            
        current_context = user_history_df.iloc[0][self.audio_cols].values
        for i in range(1, len(user_history_df)):
            track_data = user_history_df.iloc[i][self.audio_cols].values
            current_context = calculate_avalanche_context(current_context, track_data, n=5)
        return current_context
    
    def recommend(self, user_history_df, k=20, target_features=None):
        """
        Genera raccomandazioni usando Prodotto Scalare NumPy (Velocissimo).
        Accetta 'target_features' opzionale per controllo manuale (slider).
        """
        if self.df_tracks.empty or self.matrix_normalized is None:
            return pd.DataFrame(), np.zeros(9)

        # 1. PREVISIONE VETTORE TARGET
        # A. Manuale (dagli Slider)
        if target_features is not None:
            if isinstance(target_features, dict):
                 predicted_vector = np.array([target_features.get(c, 0.5) for c in self.audio_cols])
            else:
                 predicted_vector = np.array(target_features)
            predicted_vector = predicted_vector.reshape(1, -1)
        
        # B. Automatico (AI Oracle)
        elif hasattr(self, 'oracle') and self.oracle:
            current_context = self._get_current_context(user_history_df)
            predicted_vector = self.oracle.predict_target(current_context).reshape(1, -1)
        
        # C. Fallback (Media semplice)
        else:
            valid_cols = [c for c in self.audio_cols if c in user_history_df.columns]
            if valid_cols:
                predicted_vector = user_history_df[valid_cols].mean().values.reshape(1, -1)
            else:
                predicted_vector = np.array([0.5]*9).reshape(1, -1)

        # Aggiunta Rumore
        noise = np.random.normal(0, 0.02, predicted_vector.shape)
        target_vector = np.clip(predicted_vector + noise, 0, 1).astype('float32')

        # 2. CALCOLO SIMILARITÀ VETTORIALE (Core Veloce)
        # Normalizziamo il target
        target_norm_val = np.linalg.norm(target_vector)
        if target_norm_val == 0: target_norm_val = 1e-10
        target_normalized = target_vector / target_norm_val
        
        # DOT PRODUCT: Calcola la similarità per TUTTI i brani in un colpo solo
        # (N_brani, 9) @ (9, 1) -> (N_brani, 1)
        scores = np.dot(self.matrix_normalized, target_normalized.T).flatten()
        
        # Assegniamo lo score ai candidati
        # Lavoriamo su una copia dei metadati per non toccare la matrice originale
        candidates = self.df_tracks.copy()
        candidates['audio_score'] = scores

        # 3. FILTRAGGIO (Blacklist e History)
        exclude_ids = []
        if os.path.exists(self.blacklist_path):
            with open(self.blacklist_path, 'r') as f:
                exclude_ids.extend([line.strip() for line in f.readlines()])
        
        if 'id' in user_history_df.columns:
            exclude_ids.extend(user_history_df['id'].unique().tolist())
        
        # Filtro
        candidates_clean = candidates[~candidates['id'].isin(exclude_ids)]

        # Panic Mode: se abbiamo filtrato tutto, teniamo solo la history fuori
        if candidates_clean.empty:
            blacklist_only = []
            if os.path.exists(self.blacklist_path):
                 with open(self.blacklist_path, 'r') as f:
                    blacklist_only = [line.strip() for line in f.readlines()]
            candidates = candidates[~candidates['id'].isin(blacklist_only)]
        else:
            candidates = candidates_clean

        # Fallback Finale se ancora vuoto
        if candidates.empty:
            candidates = self.df_tracks.sample(n=min(k, len(self.df_tracks))).copy()
            candidates['audio_score'] = 0.5

        # 4. CALCOLO SCORE SECONDARI
        target_year = user_history_df['year'].mean() if not user_history_df.empty and 'year' in user_history_df.columns else 2022
        avg_pop = user_history_df['popularity'].mean() if not user_history_df.empty and 'popularity' in user_history_df else 50
        top_artists_set = set(user_history_df['artist'].unique()) if not user_history_df.empty and 'artist' in user_history_df else set()

        # Score Anno
        if 'year' in candidates.columns:
            candidates['year_diff'] = np.abs(candidates['year'] - target_year)
            candidates['year_score'] = 1 / (1 + (candidates['year_diff'] * 0.1))
        else:
            candidates['year_score'] = 0.5

        # Score Popolarità
        if 'popularity' in candidates.columns:
            candidates['pop_diff'] = np.abs(candidates['popularity'] - avg_pop)
            candidates['pop_score'] = 1 / (1 + (candidates['pop_diff'] * 0.05))
        else:
            candidates['pop_score'] = 0.5
        
        # Score Artista
        candidates['is_top_artist'] = False
        if top_artists_set and 'artist' in candidates.columns:
            candidates['is_top_artist'] = candidates['artist'].isin(top_artists_set)

        # 5. PESATURA FINALE
        candidates['final_score'] = (
            (candidates['audio_score'] * 0.65) + 
            (candidates['year_score'] * 0.15) + 
            (candidates['pop_score'] * 0.20)
        )
        
        if top_artists_set:
            candidates.loc[candidates['is_top_artist'], 'final_score'] *= 1.25

        # 6. SELEZIONE FINALE
        def get_reason(row):
            if row['is_top_artist']: return "DNA Artista"
            if row['audio_score'] > 0.95: return "Vibe Identica"
            if row['pop_score'] > 0.90: return "Hit Affine"
            return "Consigliato"

        sorted_candidates = candidates.sort_values(by='final_score', ascending=False)
        
        # Deduplica
        if 'name' in sorted_candidates.columns and 'artist' in sorted_candidates.columns:
            sorted_candidates['unique_key'] = sorted_candidates['name'].astype(str) + sorted_candidates['artist'].astype(str)
            sorted_candidates = sorted_candidates.drop_duplicates(subset='unique_key', keep='first')
        
        # Standard List
        standard_list = []
        artist_counts = {}
        for _, row in sorted_candidates.iterrows():
            if len(standard_list) >= (k - 5): break
            a_name = str(row.get('artist', ''))
            if artist_counts.get(a_name, 0) >= 2: continue
            
            row['reason_text'] = get_reason(row)
            row['match_percentage'] = min(row['final_score'] * 100, 99)
            standard_list.append(row)
            artist_counts[a_name] = artist_counts.get(a_name, 0) + 1
            
        final_recs = pd.DataFrame(standard_list)

        # Wildcard (Se c'è spazio)
        if len(final_recs) < k + 5:
            # Prendi brani popolari non ancora selezionati
            exclude_now = final_recs['id'].tolist() if not final_recs.empty else []
            # Usiamo il dataframe filtrato 'candidates' che ha già gli score audio calcolati
            wild_pool = candidates[
                (~candidates['id'].isin(exclude_now)) & 
                (candidates['popularity'] > 60) &
                (candidates['audio_score'].between(0.3, 0.7)) # Abbastanza diversi ma non troppo
            ]
            
            if not wild_pool.empty:
                wildcards = wild_pool.sample(n=min(5, len(wild_pool))).copy()
                wildcards['reason_text'] = "Wildcard (Novità)"
                wildcards['match_percentage'] = wildcards['audio_score'] * 100
                final_recs = pd.concat([final_recs, wildcards], ignore_index=True)

        # Shuffle finale
        if not final_recs.empty:
            final_recs = final_recs.sample(frac=1).reset_index(drop=True)
        
        return final_recs, predicted_vector.flatten()