import pandas as pd
import numpy as np
import os
import joblib
import re
from sklearn.metrics.pairwise import cosine_similarity

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
            path_attempt_1 = os.path.normpath(os.path.join(current_dir, '..', 'data', 'tracks_processed.csv'))
            path_attempt_2 = os.path.join(current_dir, 'tracks_processed.csv')
            
            if os.path.exists(path_attempt_1):
                self.tracks_path = path_attempt_1
            elif os.path.exists(path_attempt_2):
                self.tracks_path = path_attempt_2
            else:
                self.tracks_path = os.path.normpath(os.path.join(current_dir, '..', 'data', 'tracks_db.csv'))

        # 2. Gestione Percorso Oracle
        self.oracle_path = os.path.join(current_dir, '..', 'data', 'oracle.pkl')

        # 3. Caricamento Dati
        print(f"✅ Recommender: Caricamento DB da {self.tracks_path}")
        if os.path.exists(self.tracks_path):
            self.df_tracks = pd.read_csv(self.tracks_path, low_memory=False)
        else:
            print(f"⚠️ ATTENZIONE: Database non trovato in {self.tracks_path}")
            self.df_tracks = pd.DataFrame()

        # Caricamento Modello AI
        if os.path.exists(self.oracle_path):
            try:
                self.oracle = joblib.load(self.oracle_path)
            except:
                pass 
        
        self.audio_cols = ['energy', 'valence', 'danceability', 'tempo', 'loudness', 
                           'speechiness', 'acousticness', 'instrumentalness', 'liveness']
        
        self.key_map = {
            0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F', 
            6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
        }

    def _get_current_context(self, user_history_df):
        if user_history_df.empty:
            return np.array([0.5]*9)
            
        current_context = user_history_df.loc[0, self.audio_cols].values
        for i in range(1, len(user_history_df)):
            track_data = user_history_df.loc[i, self.audio_cols].values
            current_context = calculate_avalanche_context(current_context, track_data, n=5)
        return current_context
    
    def recommend(self, user_history_df, k=20):
        if self.df_tracks.empty:
            raise ValueError("Il database musicale è vuoto o non è stato caricato.")

        # 1. PREVISIONE AI
        if hasattr(self, 'oracle'):
            current_context = self._get_current_context(user_history_df)
            predicted_vector = self.oracle.predict_target(current_context).reshape(1, -1)
        else:
            valid_cols = [c for c in self.audio_cols if c in user_history_df.columns]
            if valid_cols:
                predicted_vector = user_history_df[valid_cols].mean().values.reshape(1, -1)
            else:
                predicted_vector = np.array([0.5]*9).reshape(1, -1)

        noise = np.random.normal(0, 0.05, predicted_vector.shape)
        predicted_vector = predicted_vector + noise
    
        # 2. ANALISI UTENTE
        avg_popularity = user_history_df['popularity'].mean() if 'popularity' in user_history_df else 50
        
        # Recupero Top Artists
        top_artists_raw = []
        if 'artist' in user_history_df:
            top_artists_raw = user_history_df['artist'].value_counts().head(10).index.tolist()
        
        # Creiamo un set per ricerca rapida (tutto minuscolo per evitare errori di case)
        top_artists_set = set(str(a).lower().strip() for a in top_artists_raw)
        
        target_year = 1990
        if 'year' in user_history_df.columns:
            target_year = user_history_df['year'].mean()

        # 3. FILTRAGGIO BASE
        candidates = self.df_tracks.copy()
        
        # --- NUOVO: CARICAMENTO BLACKLIST ---
        blacklist_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'blacklist.txt')
        if os.path.exists(blacklist_path):
            with open(blacklist_path, 'r') as f:
                blacklisted_ids = [line.strip() for line in f.readlines()]
            # Rimuovi i brani odiati
            if 'id' in candidates.columns:
                candidates = candidates[~candidates['id'].isin(blacklisted_ids)]
        
        if 'id' in user_history_df.columns and 'id' in candidates.columns:
            history_ids = user_history_df['id'].unique().tolist()
            candidates = candidates[~candidates['id'].isin(history_ids)]

        # 4. CALCOLO SCORE
        for col in self.audio_cols:
            if col not in candidates.columns:
                candidates[col] = 0.5 
        
        candidate_vectors = candidates[self.audio_cols].fillna(0.5)
        candidates['audio_score'] = cosine_similarity(predicted_vector, candidate_vectors)[0]
        
        if 'year' in candidates.columns:
            candidates['year'] = candidates['year'].fillna(target_year)
            candidates['year_diff'] = np.abs(candidates['year'] - target_year)
            candidates['year_score'] = 1 / (1 + (candidates['year_diff'] * 0.1))
        else:
            candidates['year_score'] = 0.5
        
        # --- FIX ARTISTA: MATCH ESATTO (Strict) ---
        candidates['is_top_artist'] = False
        
        if top_artists_set and 'artist' in candidates.columns:
            # Creiamo una colonna temporanea lower per i confronti sicuri
            candidates['artist_lower'] = candidates['artist'].astype(str).str.lower().str.strip()
            
            # SOLO Match Esatto: "Prince" == "prince". "Prince Royce" viene scartato.
            candidates['is_top_artist'] = candidates['artist_lower'].isin(top_artists_set)
            
            # Pulizia
            candidates.drop(columns=['artist_lower'], inplace=True)
            
            # Bonus
            candidates.loc[candidates['is_top_artist'], 'year_score'] = 1.0

        candidates['final_score'] = (candidates['audio_score'] * 0.70) + (candidates['year_score'] * 0.30)
        
        if top_artists_set:
            # Bonus leggermente più alto per premiare la fedeltà all'artista esatto
            candidates.loc[candidates['is_top_artist'], 'final_score'] *= 1.20

        # --- FASE 5: SELEZIONE E DE-DUPLICAZIONE ---
        def get_reason(row):
            reasons = []
            if row.get('is_top_artist'): reasons.append("Top Artista")
            if row.get('audio_score', 0) > 0.95: reasons.append("Audio Perfetto")
            elif row.get('audio_score', 0) > 0.85: reasons.append("Vibe Simile")
            
            y = row.get('year', 0)
            yd = row.get('year_diff', 0)
            if yd < 3: reasons.append(f"Anno {int(y)}")
            elif row.get('is_top_artist') and yd > 10: reasons.append("Legacy")
            elif yd < 8: reasons.append("Epoca Vicina")
            
            if row.get('popularity', 0) > 75: reasons.append("Hit")
            return " | ".join(reasons[:2])

        # A. SELEZIONE STANDARD
        sorted_candidates = candidates.sort_values(by='final_score', ascending=False)
        
        # Filtro Anti-Doppioni
        if 'name' in sorted_candidates.columns and 'artist' in sorted_candidates.columns:
            sorted_candidates['unique_key'] = (sorted_candidates['name'].astype(str).str.lower().str.split(' - ').str[0] + 
                                            sorted_candidates['artist'].astype(str).str.lower())
            sorted_candidates = sorted_candidates.drop_duplicates(subset='unique_key', keep='first')
        
        # Loop Selezione
        standard_list = []
        artist_counts = {}
        
        for idx, row in sorted_candidates.iterrows():
            artist_name = str(row.get('artist', '')).strip()
            if artist_counts.get(artist_name, 0) >= 2:
                continue
                
            row['reason_text'] = get_reason(row)
            row['match_percentage'] = np.minimum(row['final_score'] * 100, 99.9)
            standard_list.append(row)
            artist_counts[artist_name] = artist_counts.get(artist_name, 0) + 1
            
            if len(standard_list) >= (k - 5):
                break
        
        standard_recs = pd.DataFrame(standard_list)

        # B. SELEZIONE WILDCARD
        wildcards = pd.DataFrame()
        if not standard_recs.empty and 'id' in standard_recs.columns:
            exclude_ids = standard_recs['id'].tolist()
            wild_candidates = candidates[
                (~candidates['id'].isin(exclude_ids)) & 
                (candidates['popularity'] > 60) & 
                (candidates['audio_score'] < 0.6) & 
                (candidates['audio_score'] > 0.3)
            ]
            
            if not wild_candidates.empty:
                wild_candidates['unique_key'] = (wild_candidates['name'].astype(str).str.lower().str.split(' - ').str[0] + 
                                            wild_candidates['artist'].astype(str).str.lower())
                wild_candidates = wild_candidates.drop_duplicates(subset='unique_key', keep='first')
                
                sample_size = min(5, len(wild_candidates))
                wildcards = wild_candidates.sample(n=sample_size).copy()
                wildcards['reason_text'] = "Wildcard (Sorpresa)"
                wildcards['match_percentage'] = wildcards['audio_score'] * 100 

        # C. UNIONE
        final_recs = pd.concat([standard_recs, wildcards]) if not wildcards.empty else standard_recs
        
        return final_recs, predicted_vector