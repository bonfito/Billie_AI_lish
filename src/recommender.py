import pandas as pd
import numpy as np
import os
import joblib
import ast
import re
import sys 
import warnings

# Gestione warning
pd.set_option('future.no_silent_downcasting', True)
warnings.simplefilter(action='ignore', category=FutureWarning)

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from utils import calculate_avalanche_context
except ImportError:
    from src.utils import calculate_avalanche_context

def _clean_str(s):
    """Pulisce la stringa rimuovendo la sintassi delle liste Python."""
    return str(s).lower().replace('[', '').replace(']', '').replace("'", "").replace('"', '').strip()

def _make_searchable(s):
    """Crea una stringa ricercabile con virgole di guardia."""
    cleaned = _clean_str(s)
    if not cleaned or cleaned == 'nan': return ""
    # Splitta per virgola, strippa spazi e ricostruisci
    parts = [p.strip() for p in cleaned.split(',')]
    return ", " + ", ".join(parts) + ", "

class SongRecommender:
    def __init__(self, dataset_path=None):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.normpath(os.path.join(current_dir, '..', 'data'))
        
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
        self.likes_path = os.path.join(data_dir, 'liked.csv')
        self.dislikes_path = os.path.join(data_dir, 'disliked.csv')
        
        self.audio_cols = ['energy', 'valence', 'danceability', 'tempo', 'loudness', 
                           'speechiness', 'acousticness', 'instrumentalness', 'liveness']
        
        self.feedback_cols = ['id', 'name', 'artist', 'genres', 'popularity', 'year'] + self.audio_cols

        # 2. Caricamento Dati
        print(f" Recommender: Caricamento DB da {self.tracks_path}")
        if os.path.exists(self.tracks_path):
            self.df_tracks = pd.read_csv(self.tracks_path, low_memory=False)
            
            # Normalizzazione Nomi Colonne
            self.df_tracks.columns = self.df_tracks.columns.astype(str).str.lower().str.strip()
            renames = {'genre': 'genres', 'artists': 'artist', 'song': 'name', 'track_name': 'name'}
            self.df_tracks.rename(columns=renames, inplace=True)
            
            if 'artist' not in self.df_tracks.columns: self.df_tracks['artist'] = "Unknown"
            if 'genres' not in self.df_tracks.columns: self.df_tracks['genres'] = "[]"

            # --- PREPARAZIONE CAMPI DI RICERCA (STRATEGIA VIRGOLE) ---
            # Crea colonne nascoste: ", emma, " invece di "['Emma']"
            self.df_tracks['search_artist'] = self.df_tracks['artist'].apply(_make_searchable)
            self.df_tracks['search_genre'] = self.df_tracks['genres'].apply(_make_searchable)

            print(" Ottimizzazione Matrice Audio...")
            for c in self.audio_cols:
                if c not in self.df_tracks.columns: self.df_tracks[c] = 0.5
            
            self.df_tracks[self.audio_cols] = self.df_tracks[self.audio_cols].fillna(0.5).infer_objects(copy=False)
            self.matrix = self.df_tracks[self.audio_cols].values.astype('float32')
            
            norm = np.linalg.norm(self.matrix, axis=1)[:, np.newaxis]
            norm[norm == 0] = 1e-10 
            self.matrix_normalized = self.matrix / norm
            print(" Motore Audio Pronto.")
        else:
            print(f" ATTENZIONE: Database non trovato.")
            self.df_tracks = pd.DataFrame()
            self.matrix_normalized = None

        if os.path.exists(self.oracle_path):
            try: self.oracle = joblib.load(self.oracle_path)
            except: self.oracle = None
        else: self.oracle = None

    def _get_current_context(self, user_history_df):
        if user_history_df.empty: return np.array([0.5]*9)
        recent = user_history_df.tail(20).reset_index(drop=True)
        valid = [c for c in self.audio_cols if c in recent.columns]
        if not valid: return np.array([0.5]*9)

        ctx = recent.iloc[0][valid].fillna(0.5).values
        if len(ctx) < 9: ctx = np.pad(ctx, (0, 9-len(ctx)), constant_values=0.5)

        for i in range(1, len(recent)):
            track = recent.iloc[i][valid].fillna(0.5).values
            if len(track) < 9: track = np.pad(track, (0, 9-len(track)), constant_values=0.5)
            ctx = calculate_avalanche_context(ctx, track, n=5)
        return ctx
    
    def _extract_items(self, s):
        """Estrae set di elementi puliti da una stringa."""
        clean = _clean_str(s)
        if not clean: return set()
        return {x.strip() for x in clean.split(',') if x.strip() and x.strip() != 'unknown'}

    def _safe_read_csv(self, path):
        try:
            if not os.path.exists(path) or os.path.getsize(path) == 0: return pd.DataFrame()
            df = pd.read_csv(path, nrows=1, header=None)
            is_data = 'id' in str(df.iloc[0,0]).lower() or len(str(df.iloc[0,0])) > 15
            
            if not is_data:
                df = pd.read_csv(path)
                df.columns = df.columns.str.lower().str.strip()
            else:
                df = pd.read_csv(path, header=None)
                df.columns = self.feedback_cols[:df.shape[1]]
            
            renames = {'artists': 'artist', 'genre': 'genres'}
            df.rename(columns=renames, inplace=True)
            return df
        except: return pd.DataFrame()

    def _load_feedback_data(self):
        liked_artists = set()
        disliked_ids = []
        disliked_genres = set()
        disliked_artists = set()
        liked_ids = []

        try:
            df_likes = self._safe_read_csv(self.likes_path)
            if 'id' in df_likes.columns: liked_ids.extend(df_likes['id'].unique().tolist())
            if 'artist' in df_likes.columns:
                for val in df_likes['artist'].dropna():
                    liked_artists.update(self._extract_items(val))
        except: df_likes = pd.DataFrame()
        
        try:
            df_dislikes = self._safe_read_csv(self.dislikes_path)
            if 'id' in df_dislikes.columns: disliked_ids.extend(df_dislikes['id'].unique().tolist())
            if 'artist' in df_dislikes.columns:
                for val in df_dislikes['artist'].dropna():
                    disliked_artists.update(self._extract_items(val))
            if 'genres' in df_dislikes.columns:
                for g in df_dislikes['genres'].dropna(): disliked_genres.update(self._extract_items(g))
        except: pass
            
        return liked_ids, disliked_ids, disliked_genres, disliked_artists, df_likes, liked_artists

    def recommend(self, user_history_df, k=20, target_features=None, session_blacklist=None):
        if self.df_tracks.empty: return pd.DataFrame(), np.zeros(9)

        # --- PREPARAZIONE DATI STORICI ---
        if user_history_df is not None and not user_history_df.empty:
            user_history_df = user_history_df.copy()
            user_history_df.columns = user_history_df.columns.str.lower().str.strip()
            renames = {'artists': 'artist', 'genre': 'genres', 'song': 'name'}
            user_history_df.rename(columns=renames, inplace=True)
            if 'artist' not in user_history_df.columns: user_history_df['artist'] = "Unknown"
            if 'genres' not in user_history_df.columns: user_history_df['genres'] = "[]"
        else:
            user_history_df = pd.DataFrame(columns=['artist', 'genres', 'id'] + self.audio_cols)

        liked_ids, disliked_ids, disliked_genres, disliked_artists, df_likes, liked_artists = self._load_feedback_data()

        # --- GUSTI UTENTE ---
        trusted_artists = set(liked_artists)
        safe_genres = set()
        
        if 'genres' in df_likes.columns: 
            for g in df_likes['genres'].dropna(): safe_genres.update(self._extract_items(g))
        
        if not user_history_df.empty:
            recent_50 = user_history_df.tail(50)
            if 'artist' in recent_50.columns:
                for val in recent_50['artist'].dropna():
                    trusted_artists.update(self._extract_items(val))
            if 'genres' in recent_50.columns:
                for g in recent_50['genres'].dropna(): safe_genres.update(self._extract_items(g))

        # Pulizia finale
        trusted_artists = {a for a in trusted_artists if len(a) > 1}
        safe_genres = {g for g in safe_genres if len(g) > 1}

        print(f" ARTISTI FIDATI: {len(trusted_artists)} (es. {list(trusted_artists)[:3]})")
        print(f" GENERI ATTIVI: {len(safe_genres)} (es. {list(safe_genres)[:3]})")

        # --- VIBE ---
        if target_features:
            p_vec = np.array(target_features).reshape(1, -1)
        elif hasattr(self, 'oracle') and self.oracle:
            p_vec = self.oracle.predict_target(self._get_current_context(user_history_df)).reshape(1, -1)
        else:
            p_vec = np.array([0.5]*9).reshape(1, -1)

        t_norm = p_vec / (np.linalg.norm(p_vec) + 1e-10)
        scores = np.dot(self.matrix_normalized, t_norm.T).flatten()
        
        pool = self.df_tracks.copy()
        pool['audio_score'] = scores

        # --- BLACKLIST ---
        exclude = set(disliked_ids + liked_ids + (session_blacklist or []))
        if 'id' in user_history_df.columns: exclude.update(user_history_df['id'].unique())
        
        pool = pool[~pool['id'].isin(exclude)]
        
        # Blacklist Artisti (Strict)
        if disliked_artists:
            # Esclude se contiene ", artista_odiato, "
            bad_regex = r', (' + '|'.join([re.escape(a) for a in disliked_artists]) + r'), '
            pool = pool[~pool['search_artist'].str.contains(bad_regex, regex=True)]

        print(f" Pool Netto: {len(pool)}")

        final_df = pd.DataFrame()

        # --- 1. FILTRO ARTISTI  ---
        if trusted_artists:
            # Cerca ", artista, " (con virgole). Questo NON trova "emma kirkby" se cerchi "emma".
            # Usiamo un approccio a chunk per evitare regex troppo lunghe
            trusted_list = list(trusted_artists)
            chunk_size = 50 
            gold_df_list = []
            
            for i in range(0, len(trusted_list), chunk_size):
                chunk = trusted_list[i:i+chunk_size]
                # Regex: , (artist1|artist2|...), 
                pattern = r', (' + '|'.join([re.escape(a) for a in chunk]) + r'), '
                matches = pool[pool['search_artist'].str.contains(pattern, regex=True)]
                if not matches.empty:
                    gold_df_list.append(matches)
            
            if gold_df_list:
                gold_pool = pd.concat(gold_df_list).drop_duplicates()
                gold_pool = gold_pool.sort_values('audio_score', ascending=False)
                
                limit_gold = int(k * 0.7)
                gold_picks = gold_pool.head(limit_gold * 3).sample(n=min(len(gold_pool), limit_gold))
                
                def get_reason(row_s):
                    for ta in trusted_artists:
                        if f", {ta}, " in row_s: return ta
                    return "Noto"
                
                gold_picks['reason_text'] = gold_picks['search_artist'].apply(lambda x: f"Tuo Artista: {get_reason(x)}")
                final_df = pd.concat([final_df, gold_picks])
                print(f" Trovate {len(gold_picks)} canzoni dai tuoi artisti.")

        # --- 2. FILTRO GENERI ---
        slots_left = k - len(final_df)
        if slots_left > 0 and safe_genres:
            remaining_pool = pool[~pool.index.isin(final_df.index)]
            
            safe_genres_list = list(safe_genres)
            genre_df_list = []
            
            # Anche qui a chunk
            chunk_size = 50
            for i in range(0, len(safe_genres_list), chunk_size):
                chunk = safe_genres_list[i:i+chunk_size]
                pattern = r', (' + '|'.join([re.escape(g) for g in chunk]) + r'), '
                matches = remaining_pool[remaining_pool['search_genre'].str.contains(pattern, regex=True)]
                if not matches.empty:
                    genre_df_list.append(matches)
            
            if genre_df_list:
                silver_pool = pd.concat(genre_df_list).drop_duplicates()
                silver_pool = silver_pool.sort_values('audio_score', ascending=False)
                
                silver_picks = silver_pool.head(slots_left * 5).sample(n=min(len(silver_pool), slots_left))
                
                def get_gen_reason(row_s):
                    for tg in safe_genres:
                        if f", {tg}, " in row_s: return tg
                    return "Genere"

                silver_picks['reason_text'] = silver_picks['search_genre'].apply(lambda x: f"Genere: {get_gen_reason(x)}")
                final_df = pd.concat([final_df, silver_picks])
                print(f" Trovate {len(silver_picks)} canzoni per genere.")

        # --- 3. FALLBACK AUDIO ---
        if len(final_df) < k:
            needed = k - len(final_df)
            remaining = pool[~pool.index.isin(final_df.index)].sort_values('audio_score', ascending=False).head(needed)
            if not remaining.empty:
                remaining['reason_text'] = "Solo Audio (Fallback)"
                final_df = pd.concat([final_df, remaining])

        if not final_df.empty:
            final_df['match_percentage'] = (final_df['audio_score'] * 100).astype(int)
            final_df = final_df.sample(frac=1).reset_index(drop=True).head(k)

        print("\n" + "="*50)
        print(" ANTEPRIMA GENERAZIONE:")
        if not final_df.empty:
            for idx, row in final_df.iterrows():
                print(f"{idx+1}. [{row.get('reason_text', '?')}] {row['artist']} - {row['name']}")
        print("="*50 + "\n")

        return final_df, p_vec.flatten()