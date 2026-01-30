import pandas as pd
import numpy as np
import os
import joblib
import ast
import re

# --- IMPORT INTELLIGENTE ---
try:
    from utils import calculate_avalanche_context
except ImportError:
    from src.utils import calculate_avalanche_context

class SongRecommender:
    def __init__(self, dataset_path=None):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 1. Configurazione Percorsi
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

        self.oracle_path = os.path.join(current_dir, '..', 'data', 'oracle.pkl')
        
        self.audio_cols = ['energy', 'valence', 'danceability', 'tempo', 'loudness', 
                           'speechiness', 'acousticness', 'instrumentalness', 'liveness']

        # 2. Caricamento Dati
        print(f"✅ Recommender: Caricamento DB da {self.tracks_path}")
        if os.path.exists(self.tracks_path):
            self.df_tracks = pd.read_csv(self.tracks_path, low_memory=False)
            print("⚙️ Ottimizzazione Matrice Audio...")
            self.matrix = self.df_tracks[self.audio_cols].fillna(0.5).values.astype('float32')
            
            norm = np.linalg.norm(self.matrix, axis=1)[:, np.newaxis]
            norm[norm == 0] = 1e-10 
            self.matrix_normalized = self.matrix / norm
            print("✅ Motore Audio Pronto.")
        else:
            print(f"⚠️ ATTENZIONE: Database non trovato.")
            self.df_tracks = pd.DataFrame()
            self.matrix_normalized = None

        # 3. Caricamento Oracle
        if os.path.exists(self.oracle_path):
            try: self.oracle = joblib.load(self.oracle_path)
            except: self.oracle = None
        else: self.oracle = None

    def _get_current_context(self, user_history_df):
        if user_history_df.empty: return np.array([0.5]*9)
        recent_history = user_history_df.tail(20).reset_index(drop=True)
        current_context = recent_history.iloc[0][self.audio_cols].values
        for i in range(1, len(recent_history)):
            track_data = recent_history.iloc[i][self.audio_cols].values
            current_context = calculate_avalanche_context(current_context, track_data, n=5)
        return current_context
    
    def _extract_genres(self, genre_str):
        if pd.isna(genre_str) or str(genre_str) == '[]': return set()
        raw_set = set()
        try:
            if str(genre_str).strip().startswith('['):
                raw_list = ast.literal_eval(genre_str)
                if isinstance(raw_list, list):
                    for g in raw_list: raw_set.add(str(g).lower())
            else:
                for g in str(genre_str).split(','):
                    raw_set.add(g.strip().lower())
        except: return set()
            
        expanded_set = set(raw_set)
        for g in raw_set:
            words = g.split()
            if len(words) > 1: expanded_set.update(words)
        return expanded_set

    def recommend(self, user_history_df, k=20, target_features=None, session_blacklist=None):
        if self.df_tracks.empty: return pd.DataFrame(), np.zeros(9)

        # --- STEP 0: DEFINISCI CHI ESCLUDERE ---
        exclude_ids = []
        if session_blacklist: exclude_ids.extend(session_blacklist)
        if 'id' in user_history_df.columns: exclude_ids.extend(user_history_df['id'].unique().tolist())
        
        # --- STEP 1: LOGICA "I MIEI ARTISTI" (Priorità) ---
        my_artists = set()
        if not user_history_df.empty:
            recent_50 = user_history_df.tail(50)
            if 'artist' in recent_50.columns:
                my_artists = set(recent_50['artist'].unique())

        familiar_recs = pd.DataFrame()
        
        if my_artists:
            artist_pool = self.df_tracks[
                (self.df_tracks['artist'].isin(my_artists)) & 
                (~self.df_tracks['id'].isin(exclude_ids))
            ].copy()
            
            if not artist_pool.empty:
                # Ordine: Prima le più recenti
                artist_pool = artist_pool.sort_values(by='year', ascending=False)
                
                # Prendiamo fino al 60% dei posti
                limit_familiar = int(k * 0.6) 
                familiar_recs = artist_pool.head(limit_familiar).copy()
                
                familiar_recs['reason_text'] = "Tuo Artista (Nuova)"
                familiar_recs['match_percentage'] = 100 
                familiar_recs['audio_score'] = 1.0 

        # --- STEP 2: LOGICA "SCOPERTA" (Vector Search) ---
        if target_features is not None:
            if isinstance(target_features, dict):
                 predicted_vector = np.array([target_features.get(c, 0.5) for c in self.audio_cols])
            else:
                 predicted_vector = np.array(target_features)
            predicted_vector = predicted_vector.reshape(1, -1)
        elif hasattr(self, 'oracle') and self.oracle:
            current_context = self._get_current_context(user_history_df)
            predicted_vector = self.oracle.predict_target(current_context).reshape(1, -1)
        else:
            valid = [c for c in self.audio_cols if c in user_history_df.columns]
            if valid: predicted_vector = user_history_df[valid].mean().values.reshape(1, -1)
            else: predicted_vector = np.array([0.5]*9).reshape(1, -1)

        noise = np.random.normal(0, 0.02, predicted_vector.shape)
        target_vector = np.clip(predicted_vector + noise, 0, 1).astype('float32')
        target_norm = np.linalg.norm(target_vector)
        if target_norm == 0: target_norm = 1e-10
        target_normalized = target_vector / target_norm
        
        scores = np.dot(self.matrix_normalized, target_normalized.T).flatten()
        
        discovery_pool = self.df_tracks.copy()
        discovery_pool['audio_score'] = scores

        user_genres = set()
        if 'genres' in user_history_df.columns:
            for g_str in user_history_df['genres'].dropna():
                user_genres.update(self._extract_genres(g_str))
        
        ids_already_picked = familiar_recs['id'].tolist() if not familiar_recs.empty else []
        ids_to_exclude_discovery = exclude_ids + ids_already_picked
        
        discovery_pool = discovery_pool[~discovery_pool['id'].isin(ids_to_exclude_discovery)]
        
        if user_genres:
            def has_common_genre(row_g):
                g_set = self._extract_genres(row_g)
                return not g_set.isdisjoint(user_genres) if g_set else False
            
            top_idx = np.argsort(discovery_pool['audio_score'].values)[::-1][:10000]
            subset = discovery_pool.iloc[top_idx].copy()
            valid_mask = subset['genres'].apply(has_common_genre)
            discovery_final = subset[valid_mask].copy()
        else:
            discovery_final = discovery_pool.sort_values(by='audio_score', ascending=False).head(5000)

        target_year = user_history_df['year'].mean() if 'year' in user_history_df.columns else 2022
        if 'year' in discovery_final.columns:
            discovery_final['year_score'] = 1 / (1 + (np.abs(discovery_final['year'] - target_year) * 0.1))
        else: discovery_final['year_score'] = 0.5

        discovery_final['final_score'] = (
            (discovery_final['audio_score'] * 0.7) + 
            (discovery_final['year_score'] * 0.3)
        )
        
        discovery_final = discovery_final.sort_values(by='final_score', ascending=False)
        
        if 'name' in discovery_final.columns and 'artist' in discovery_final.columns:
            discovery_final['unique_key'] = discovery_final['name'].astype(str) + discovery_final['artist'].astype(str)
            discovery_final = discovery_final.drop_duplicates(subset='unique_key', keep='first')

        slots_needed = k - len(familiar_recs)
        discovery_recs = pd.DataFrame()
        
        if slots_needed > 0:
            temp_list = []
            a_counts = {}
            for _, row in discovery_final.iterrows():
                if len(temp_list) >= slots_needed: break
                a_name = str(row.get('artist', ''))
                if a_counts.get(a_name, 0) >= 1: continue 
                
                row['reason_text'] = "Scoperta Simile"
                row['match_percentage'] = min(row['final_score'] * 100, 99)
                temp_list.append(row)
                a_counts[a_name] = a_counts.get(a_name, 0) + 1
            
            discovery_recs = pd.DataFrame(temp_list)

        # --- STEP 3: UNIONE & SHUFFLE INTELLIGENTE ---
        # 1. Teniamo le prime 5 "Familiari" fisse in cima (Garanzia di qualità iniziale)
        # 2. Mischiamo il resto (Familiari rimanenti + Scoperte)
        
        top_picks = pd.DataFrame()
        mix_pool = pd.DataFrame()

        if not familiar_recs.empty:
            top_picks = familiar_recs.head(5) # Le prime 5 sono sacre
            rest_familiar = familiar_recs.iloc[5:]
            mix_pool = pd.concat([rest_familiar, discovery_recs], ignore_index=True)
        else:
            # Se non ci sono familiari, mischiamo tutto le scoperte tranne la prima
            top_picks = discovery_recs.head(1)
            mix_pool = discovery_recs.iloc[1:]

        # Mischiamo il pool rimanente per evitare il blocco "solo scoperte" alla fine
        if not mix_pool.empty:
            mix_pool = mix_pool.sample(frac=1).reset_index(drop=True)

        final_df = pd.concat([top_picks, mix_pool], ignore_index=True)
        
        if final_df.empty:
            final_df = self.df_tracks.sample(n=k).copy()
            final_df['reason_text'] = "Fallback"
            final_df['match_percentage'] = 50

        return final_df, predicted_vector.flatten()