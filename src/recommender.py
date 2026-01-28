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

        # Caricamento Modello AI Oracle
        if os.path.exists(self.oracle_path):
            try:
                self.oracle = joblib.load(self.oracle_path)
            except:
                pass 
        
        self.audio_cols = ['energy', 'valence', 'danceability', 'tempo', 'loudness', 
                           'speechiness', 'acousticness', 'instrumentalness', 'liveness']

    def _get_current_context(self, user_history_df):
        """Calcola il vettore medio pesato (Avalanche) della storia utente."""
        if user_history_df.empty:
            return np.array([0.5]*9)
            
        current_context = user_history_df.iloc[0][self.audio_cols].values
        for i in range(1, len(user_history_df)):
            track_data = user_history_df.iloc[i][self.audio_cols].values
            current_context = calculate_avalanche_context(current_context, track_data, n=5)
        return current_context
    
    def recommend(self, user_history_df, k=20):
        """Genera raccomandazioni basate su DNA audio, Popolarità e Case-Sensitive Artist match."""
        if self.df_tracks.empty:
            raise ValueError("Il database musicale è vuoto.")

        # 1. PREVISIONE VETTORE TARGET (AI ORACLE)
        if hasattr(self, 'oracle'):
            current_context = self._get_current_context(user_history_df)
            predicted_vector = self.oracle.predict_target(current_context).reshape(1, -1)
        else:
            valid_cols = [c for c in self.audio_cols if c in user_history_df.columns]
            if valid_cols:
                predicted_vector = user_history_df[valid_cols].mean().values.reshape(1, -1)
            else:
                predicted_vector = np.array([0.5]*9).reshape(1, -1)

        # Aggiunta di un leggero rumore per evitare raccomandazioni statiche
        noise = np.random.normal(0, 0.02, predicted_vector.shape)
        predicted_vector = np.clip(predicted_vector + noise, 0, 1)

        # 2. ANALISI PROFILO UTENTE (STRICT CASE)
        top_artists_set = set(user_history_df['artist'].unique()) if 'artist' in user_history_df else set()
        avg_pop = user_history_df['popularity'].mean() if 'popularity' in user_history_df else 50
        target_year = user_history_df['year'].mean() if 'year' in user_history_df.columns else 2010

        # 3. FILTRAGGIO (Esclusioni)
        candidates = self.df_tracks.copy()
        
        # Blacklist (brani skippati)
        blacklist_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'blacklist.txt')
        exclude_ids = []
        if os.path.exists(blacklist_path):
            with open(blacklist_path, 'r') as f:
                exclude_ids.extend([line.strip() for line in f.readlines()])
        
        # Escludi brani già in cronologia
        if 'id' in user_history_df.columns:
            exclude_ids.extend(user_history_df['id'].unique().tolist())
        
        candidates = candidates[~candidates['id'].isin(exclude_ids)]

        # 4. CALCOLO SCORE MULTI-FATTORE
        # A. Audio Similarity
        candidate_vectors = candidates[self.audio_cols].fillna(0.5).values
        candidates['audio_score'] = cosine_similarity(predicted_vector, candidate_vectors)[0]
        
        # B. Temporal Score (vicinanza anni)
        if 'year' in candidates.columns:
            candidates['year_diff'] = np.abs(candidates['year'] - target_year)
            candidates['year_score'] = 1 / (1 + (candidates['year_diff'] * 0.1))
        else:
            candidates['year_score'] = 0.5

        # C. Popularity Score (per evitare omonimi "fake" a popolarità 0)
        if 'popularity' in candidates.columns:
            candidates['pop_diff'] = np.abs(candidates['popularity'] - avg_pop)
            candidates['pop_score'] = 1 / (1 + (candidates['pop_diff'] * 0.05))
        else:
            candidates['pop_score'] = 0.5
        
        # D. Strict Artist Match (Case Sensitive)
        candidates['is_top_artist'] = False
        if top_artists_set and 'artist' in candidates.columns:
            candidates['is_top_artist'] = candidates['artist'].isin(top_artists_set)

        # 5. PESATURA FINALE
        candidates['final_score'] = (
            (candidates['audio_score'] * 0.60) + 
            (candidates['year_score'] * 0.20) + 
            (candidates['pop_score'] * 0.20)
        )
        
        # Bonus drastico per Top Artist (se scritto esattamente uguale)
        if top_artists_set:
            candidates.loc[candidates['is_top_artist'], 'final_score'] *= 1.30

        # 6. SELEZIONE E DE-DUPLICAZIONE (Logica Avanzata)
        def get_reason(row):
            if row['is_top_artist']: return "DNA Artista"
            if row['audio_score'] > 0.96: return "Vibe Identica"
            if row['pop_score'] > 0.90 and row['audio_score'] > 0.85: return "Hit Affine"
            return "Scoperta AI"

        sorted_candidates = candidates.sort_values(by='final_score', ascending=False)
        
        # Unique Key Case Sensitive per evitare versioni live/remix identiche
        sorted_candidates['unique_key'] = sorted_candidates['name'].astype(str) + sorted_candidates['artist'].astype(str)
        sorted_candidates = sorted_candidates.drop_duplicates(subset='unique_key', keep='first')
        
        # --- A. SELEZIONE STANDARD (k-5 brani) ---
        standard_list = []
        artist_counts = {}
        
        for idx, row in sorted_candidates.iterrows():
            a_name = str(row.get('artist', ''))
            # Limita max 2 brani dello stesso artista per sessione
            if artist_counts.get(a_name, 0) >= 2:
                continue
                
            row['reason_text'] = get_reason(row)
            row['match_percentage'] = np.minimum(row['final_score'] * 100, 99.9)
            standard_list.append(row)
            artist_counts[a_name] = artist_counts.get(a_name, 0) + 1
            
            if len(standard_list) >= (k - 5):
                break
        
        standard_recs = pd.DataFrame(standard_list)

        # --- B. SELEZIONE WILDCARD (5 brani "Sorpresa") ---
        # Brani con audio score medio ma alta popolarità (scoperte laterali)
        exclude_now = standard_recs['id'].tolist() if not standard_recs.empty else []
        wild_candidates = candidates[
            (~candidates['id'].isin(exclude_now)) & 
            (candidates['popularity'] > 65) & 
            (candidates['audio_score'].between(0.4, 0.7))
        ]
        
        wildcards = pd.DataFrame()
        if not wild_candidates.empty:
            wildcards = wild_candidates.sample(n=min(5, len(wild_candidates))).copy()
            wildcards['reason_text'] = "Wildcard (Novità)"
            wildcards['match_percentage'] = wildcards['audio_score'] * 100

        # --- C. UNIONE FINALE ---
        final_recs = pd.concat([standard_recs, wildcards], ignore_index=True) if not wildcards.empty else standard_recs
        
        # Shuffle finale per non avere sempre le wildcard in fondo
        final_recs = final_recs.sample(frac=1).reset_index(drop=True)
        
        return final_recs, predicted_vector