import pandas as pd
import numpy as np
import os
import joblib
import sqlite3
import faiss

try:
    from utils import calculate_avalanche_context
except ImportError:
    def calculate_avalanche_context(current, new_track, n=5):
        return (current * 0.9) + (new_track * 0.1)

class SongRecommender:
    def __init__(self, dataset_path=None):
        # Nota: dataset_path è mantenuto per compatibilità ma ignorato a favore di DB/Index
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.normpath(os.path.join(current_dir, '..', 'data'))
        
        self.db_path = os.path.join(data_dir, 'tracks.db')
        self.index_path = os.path.join(data_dir, 'tracks.index')
        self.oracle_path = os.path.join(data_dir, 'oracle.pkl')
        self.blacklist_path = os.path.join(data_dir, 'blacklist.txt')

        # Caricamento FAISS
        print(f"✅ Recommender: Caricamento FAISS da {self.index_path}")
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            print(f"⚠️ Indice FAISS non trovato. Esegui build_index.py")
            self.index = None

        # Caricamento Oracle
        if os.path.exists(self.oracle_path):
            try: self.oracle = joblib.load(self.oracle_path)
            except: pass
        else: self.oracle = None
        
        self.audio_cols = ['energy', 'valence', 'danceability', 'tempo', 'loudness', 
                           'speechiness', 'acousticness', 'instrumentalness', 'liveness']

    def _get_current_context(self, user_history_df):
        if user_history_df.empty: return np.array([0.5]*9)
        current_context = user_history_df.iloc[0][self.audio_cols].values
        # Usiamo avalanche se ci sono abbastanza brani, altrimenti media
        if len(user_history_df) > 1:
            for i in range(1, min(len(user_history_df), 20)): # Limitiamo a 20 per velocità
                track_data = user_history_df.iloc[i][self.audio_cols].values
                current_context = calculate_avalanche_context(current_context, track_data, n=5)
        return current_context

    def _query_faiss(self, vector, k=100):
        """Esegue ricerca Cosine Similarity tramite FAISS (Inner Product su vettori normalizzati)"""
        if self.index is None: return [], []
        
        # 1. Preparazione Vettore
        vector = np.ascontiguousarray(vector.astype('float32').reshape(1, -1))
        
        # 2. Normalizzazione L2 (Cruciale per Cosine Similarity)
        faiss.normalize_L2(vector)
        
        # 3. Ricerca
        distances, indices = self.index.search(vector, k)
        return distances[0], indices[0]

    def _fetch_candidates(self, faiss_ids):
        """Recupera metadati da SQLite"""
        if len(faiss_ids) == 0: return pd.DataFrame()
        conn = sqlite3.connect(self.db_path)
        ids_str = ','.join(map(str, faiss_ids))
        query = f"SELECT * FROM tracks WHERE faiss_id IN ({ids_str})"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    def recommend(self, user_history_df, k=20, target_features=None):
        if self.index is None: return pd.DataFrame(), np.zeros(9)

        # 1. CALCOLO VETTORE TARGET
        if target_features is not None:
             # Manuale (Slider)
            if isinstance(target_features, dict):
                 predicted_vector = np.array([target_features.get(c, 0.5) for c in self.audio_cols])
            else:
                 predicted_vector = np.array(target_features)
        elif hasattr(self, 'oracle') and self.oracle:
            # AI Oracle
            ctx = self._get_current_context(user_history_df)
            predicted_vector = self.oracle.predict_target(ctx)
        else:
            # Fallback Media
            valid = [c for c in self.audio_cols if c in user_history_df.columns]
            if valid: predicted_vector = user_history_df[valid].mean().values
            else: predicted_vector = np.array([0.5]*9)
            
        # Aggiunta Rumore
        noise = np.random.normal(0, 0.02, predicted_vector.shape)
        search_vector = np.clip(predicted_vector + noise, 0, 1)

        # 2. RICERCA CANDIDATI (FAISS)
        # Cerchiamo K*50 candidati (es. 1000 brani) per avere margine sui filtri
        scores, faiss_ids = self._query_faiss(search_vector, k=k*50)
        
        # Recupero Metadati
        candidates = self._fetch_candidates(faiss_ids)
        if candidates.empty: return pd.DataFrame(), predicted_vector.flatten()

        # Mappa Score Audio
        score_map = dict(zip(faiss_ids, scores))
        candidates['audio_score'] = candidates['faiss_id'].map(score_map)

        # 3. FILTRI (Blacklist)
        exclude_ids = []
        if os.path.exists(self.blacklist_path):
            with open(self.blacklist_path, 'r') as f: exclude_ids.extend([l.strip() for l in f])
        if 'id' in user_history_df.columns: exclude_ids.extend(user_history_df['id'].unique())
        
        # Nota: nel DB SQLite la colonna id originale è stata rinominata in spotify_id
        candidates = candidates[~candidates['spotify_id'].isin(exclude_ids)]

        # 4. CALCOLO SCORE SECONADRI (Logica Originale Mantenuta)
        target_year = user_history_df['year'].mean() if 'year' in user_history_df.columns else 2022
        avg_pop = user_history_df['popularity'].mean() if 'popularity' in user_history_df.columns else 50
        top_artists = set(user_history_df['artist'].unique()) if 'artist' in user_history_df.columns else set()

        if 'year' in candidates.columns:
            candidates['year_diff'] = np.abs(candidates['year'] - target_year)
            candidates['year_score'] = 1 / (1 + (candidates['year_diff'] * 0.1))
        else: candidates['year_score'] = 0.5

        if 'popularity' in candidates.columns:
            candidates['pop_diff'] = np.abs(candidates['popularity'] - avg_pop)
            candidates['pop_score'] = 1 / (1 + (candidates['pop_diff'] * 0.05))
        else: candidates['pop_score'] = 0.5
        
        candidates['is_top_artist'] = False
        if 'artist' in candidates.columns:
            candidates['is_top_artist'] = candidates['artist'].isin(top_artists)

        # 5. SCORE FINALE
        candidates['final_score'] = (
            (candidates['audio_score'] * 0.60) + 
            (candidates['year_score'] * 0.20) + 
            (candidates['pop_score'] * 0.20)
        )
        
        if top_artists:
            candidates.loc[candidates['is_top_artist'], 'final_score'] *= 1.30

        # 6. SELEZIONE (Standard + Wildcard)
        sorted_candidates = candidates.sort_values(by='final_score', ascending=False)
        
        # Deduplica
        if 'name' in sorted_candidates.columns and 'artist' in sorted_candidates.columns:
            sorted_candidates['unique_key'] = sorted_candidates['name'].astype(str) + sorted_candidates['artist'].astype(str)
            sorted_candidates = sorted_candidates.drop_duplicates(subset='unique_key', keep='first')
        
        # -- Standard List --
        standard_list = []
        artist_counts = {}
        for _, row in sorted_candidates.iterrows():
            if len(standard_list) >= (k - 5): break
            a_name = str(row.get('artist', ''))
            if artist_counts.get(a_name, 0) >= 2: continue
            
            # Rinomina spotify_id -> id per coerenza con app
            row_dict = row.to_dict()
            row_dict['id'] = row_dict.pop('spotify_id')
            
            if row['is_top_artist']: reason = "DNA Artista"
            elif row['audio_score'] > 0.96: reason = "Vibe Identica"
            elif row['pop_score'] > 0.90: reason = "Hit Affine"
            else: reason = "Scoperta AI"
            
            row_dict['reason_text'] = reason
            row_dict['match_percentage'] = min(row['final_score'] * 100, 99)
            standard_list.append(row_dict)
            artist_counts[a_name] = artist_counts.get(a_name, 0) + 1
            
        standard_recs = pd.DataFrame(standard_list)

        # -- Wildcard (5 brani popolari ma audio-compatibili) --
        # Prendiamo dal pool dei candidati rimanenti
        exclude_now = standard_recs['id'].tolist() if not standard_recs.empty else []
        # Filtriamo candidati non usati
        remaining = candidates[~candidates['spotify_id'].isin(exclude_now)]
        
        wild_candidates = remaining[
            (remaining['popularity'] > 65) & 
            (remaining['audio_score'].between(0.4, 0.7)) # Audio diverso ma non troppo
        ]
        
        wildcards = pd.DataFrame()
        if not wild_candidates.empty:
            wildcards = wild_candidates.sample(n=min(5, len(wild_candidates))).copy()
            wildcards = wildcards.rename(columns={'spotify_id': 'id'})
            wildcards['reason_text'] = "Wildcard (Novità)"
            wildcards['match_percentage'] = wildcards['audio_score'] * 100

        # Unione
        final_recs = pd.concat([standard_recs, wildcards], ignore_index=True) if not wildcards.empty else standard_recs
        if not final_recs.empty:
            final_recs = final_recs.sample(frac=1).reset_index(drop=True)

        return final_recs, predicted_vector.flatten()