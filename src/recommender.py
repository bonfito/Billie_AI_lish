import pandas as pd
import numpy as np
import os
import joblib
import ast
import re
import sys 

# --- IMPORT INTELLIGENTE ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from utils import calculate_avalanche_context
except ImportError:
    from src.utils import calculate_avalanche_context

class SongRecommender:
    def __init__(self, dataset_path=None):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.normpath(os.path.join(current_dir, '..', 'data'))
        
        # 1. Configurazione Percorsi
        if dataset_path:
            self.tracks_path = dataset_path
        else:
            path_attempt_1 = os.path.join(data_dir, 'tracks_processed.csv')
            path_attempt_2 = os.path.join(current_dir, 'tracks_processed.csv')
            self.tracks_path = path_attempt_1 if os.path.exists(path_attempt_1) else os.path.normpath(os.path.join(data_dir, 'tracks_db.csv'))

        self.oracle_path = os.path.join(data_dir, 'oracle.pkl')
        
        # Feedback persistente
        self.likes_path = os.path.join(data_dir, 'likes.csv')
        self.dislikes_path = os.path.join(data_dir, 'dislikes.csv')
        
        self.audio_cols = ['energy', 'valence', 'danceability', 'tempo', 'loudness', 
                           'speechiness', 'acousticness', 'instrumentalness', 'liveness']

        # 2. Caricamento Dati
        print(f"✅ Recommender: Caricamento DB da {self.tracks_path}")
        if os.path.exists(self.tracks_path):
            self.df_tracks = pd.read_csv(self.tracks_path, low_memory=False)
            
            # --- FIX COLONNE ---
            if 'artists' in self.df_tracks.columns and 'artist' not in self.df_tracks.columns:
                self.df_tracks.rename(columns={'artists': 'artist'}, inplace=True)
            if 'artist' not in self.df_tracks.columns:
                self.df_tracks['artist'] = "Unknown Artist"
            for c in self.audio_cols:
                if c not in self.df_tracks.columns: self.df_tracks[c] = 0.5
            
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
        recent = user_history_df.tail(20).reset_index(drop=True)
        valid = [c for c in self.audio_cols if c in recent.columns]
        if not valid: return np.array([0.5]*9)

        ctx = recent.iloc[0][self.audio_cols].fillna(0.5).values
        for i in range(1, len(recent)):
            track = recent.iloc[i][self.audio_cols].fillna(0.5).values
            ctx = calculate_avalanche_context(ctx, track, n=5)
        return ctx
    
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
        
        expanded = set(raw_set)
        for g in raw_set:
            words = g.split()
            if len(words) > 1: expanded.update(words)
        return expanded

    def _load_feedback_data(self):
        liked_artists = set()
        disliked_ids = set()
        disliked_genres = set()

        if os.path.exists(self.likes_path):
            try:
                df_likes = pd.read_csv(self.likes_path)
                if 'artist' in df_likes.columns:
                    liked_artists.update(df_likes['artist'].unique())
            except: pass

        if os.path.exists(self.dislikes_path):
            try:
                df_dislikes = pd.read_csv(self.dislikes_path)
                if 'id' in df_dislikes.columns:
                    disliked_ids.update(df_dislikes['id'].unique())
                
                col_g = 'genres' if 'genres' in df_dislikes.columns else 'genre'
                if col_g in df_dislikes.columns:
                    for g_str in df_dislikes[col_g].dropna():
                        disliked_genres.update(self._extract_genres(g_str))
            except: pass
            
        return liked_artists, disliked_ids, disliked_genres

    def recommend(self, user_history_df, k=20, target_features=None, session_blacklist=None):
        """
        LOGICA STRICT:
        1. Priorità Totale ad Artisti conosciuti (70% dei risultati).
        2. Filtro Generi Severo (No scoperte random).
        3. Shuffle solo tra i "buoni" candidati.
        """
        if self.df_tracks.empty: return pd.DataFrame(), np.zeros(9)

        # Carica Feedback
        liked_artists_csv, disliked_ids_csv, disliked_genres_csv = self._load_feedback_data()

        # --- STEP 0: BLACKLIST TOTALE ---
        exclude_ids = []
        if session_blacklist: exclude_ids.extend(session_blacklist)
        if 'id' in user_history_df.columns: exclude_ids.extend(user_history_df['id'].unique().tolist())
        exclude_ids.extend(list(disliked_ids_csv))

        # --- PREPARAZIONE CONTESTO ---
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

        # Riduciamo il rumore quasi a zero per evitare brani "sbagliati"
        noise = np.random.normal(0, 0.02, predicted_vector.shape) 
        target_vector = np.clip(predicted_vector + noise, 0, 1).astype('float32')
        target_norm = np.linalg.norm(target_vector)
        if target_norm == 0: target_norm = 1e-10
        target_normalized = target_vector / target_norm
        
        # Calcolo Score Audio
        scores = np.dot(self.matrix_normalized, target_normalized.T).flatten()
        
        pool = self.df_tracks.copy()
        pool['audio_score'] = scores
        pool = pool[~pool['id'].isin(exclude_ids)]

        # --- ANALISI GENERI UTENTE ---
        user_genres = set()
        # Prendi generi da history
        genre_col_hist = 'genres' if 'genres' in user_history_df.columns else ('genre' if 'genre' in user_history_df.columns else None)
        if genre_col_hist:
            for g_str in user_history_df[genre_col_hist].dropna():
                user_genres.update(self._extract_genres(g_str))
        
        # --- STEP 1: I MIEI ARTISTI (70% dei risultati) ---
        my_artists = set()
        if not user_history_df.empty:
            recent_50 = user_history_df.tail(50)
            h_artist_col = 'artist' if 'artist' in recent_50.columns else ('artists' if 'artists' in recent_50.columns else None)
            if h_artist_col:
                my_artists = set(recent_50[h_artist_col].unique())
        my_artists.update(liked_artists_csv)

        familiar_recs = pd.DataFrame()
        if my_artists:
            artist_pool = pool[pool['artist'].isin(my_artists)].copy()
            
            if not artist_pool.empty:
                # Prendiamo il 70% dei posti (k * 0.7)
                limit_familiar = int(k * 0.7) 
                
                # Per variare ma restare "corretti", prendiamo dai top 30% per score audio
                # (così sono artisti tuoi E suonano simili al mood attuale)
                quantile_threshold = artist_pool['audio_score'].quantile(0.7) # Top 30%
                good_candidates = artist_pool[artist_pool['audio_score'] >= quantile_threshold]
                
                if good_candidates.empty: good_candidates = artist_pool
                
                # Pesca casuale tra i candidati OTTIMI del tuo artista
                if len(good_candidates) > limit_familiar:
                    familiar_recs = good_candidates.sample(n=limit_familiar).copy()
                else:
                    familiar_recs = good_candidates.copy()

                familiar_recs['reason_text'] = "Tuo Artista"
                familiar_recs['match_percentage'] = 100

        # --- STEP 2: STESSO GENERE (30% dei risultati) ---
        genre_recs = pd.DataFrame()
        slots_needed = k - len(familiar_recs)
        
        if slots_needed > 0 and user_genres and 'genres' in pool.columns:
            
            # Filtro Cecchino: SOLO se ha generi in comune con i tuoi
            def is_strict_genre(row_g):
                g_set = self._extract_genres(row_g)
                if not g_set: return False
                # BANNED se tocca un genere odiato
                if not g_set.isdisjoint(disliked_genres_csv): return False
                # ACCEPTED solo se tocca un genere amato
                if not g_set.isdisjoint(user_genres): return True
                return False 

            # Applichiamo il filtro su un bacino più ampio ma ordinato per audio
            top_audio_candidates = pool.sort_values(by='audio_score', ascending=False).head(15000)
            
            safe_candidates = top_audio_candidates[top_audio_candidates['genres'].apply(is_strict_genre)].copy()
            
            if not safe_candidates.empty:
                # Anche qui, pesca casuale dai top per variare al refresh
                pool_size = slots_needed * 5 # Pesca da un pool 5 volte più grande
                best_genre_pool = safe_candidates.head(pool_size)
                
                genre_recs = best_genre_pool.sample(n=min(len(best_genre_pool), slots_needed)).copy()
                genre_recs['reason_text'] = "Genere Simile"
                genre_recs['match_percentage'] = (genre_recs['audio_score'] * 100).astype(int)
        
        # Fallback (solo se proprio non trova nulla)
        if genre_recs.empty and slots_needed > 0:
            remaining_pool = pool.sort_values(by='audio_score', ascending=False).head(slots_needed * 2)
            genre_recs = remaining_pool.sample(n=min(len(remaining_pool), slots_needed)).copy()
            genre_recs['reason_text'] = "Mix Audio"
            genre_recs['match_percentage'] = (genre_recs['audio_score'] * 90).astype(int)

        # --- STEP 3: UNIONE E SHUFFLE FINALE ---
        final_df = pd.concat([familiar_recs, genre_recs], ignore_index=True)
        
        # Dedup finale
        if 'name' in final_df.columns and 'artist' in final_df.columns:
            final_df['unique_key'] = final_df['name'].astype(str) + final_df['artist'].astype(str)
            final_df = final_df.drop_duplicates(subset='unique_key', keep='first')
        
        # SHUFFLE FINALE: Mischiamo tutto per non avere blocchi "Solo Artista" poi "Solo Genere"
        if not final_df.empty:
            final_df = final_df.sample(frac=1).reset_index(drop=True)
            final_df = final_df.head(k)
        else:
            final_df = self.df_tracks.sample(n=k).copy()
            final_df['reason_text'] = "Random Fallback"
            final_df['match_percentage'] = 0

        return final_df, predicted_vector.flatten()