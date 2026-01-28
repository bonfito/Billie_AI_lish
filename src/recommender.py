import pandas as pd
import numpy as np
import os
import joblib
import sqlite3
import faiss

# import utils
try:
    from utils import calculate_weighted_context
except ImportError:
    # Se calculate_weighted_context non esiste ancora in utils, definiamo un fallback locale
    def calculate_weighted_context(history_df, audio_cols):
        # Media ponderata dei vettori usando la colonna 'weight'
        if history_df.empty or 'weight' not in history_df.columns:
            return np.array([0.5] * len(audio_cols))
        
        vectors = history_df[audio_cols].values
        weights = history_df['weight'].values.reshape(-1, 1)
        
        # Normalizziamo i pesi
        total_weight = np.sum(weights)
        if total_weight == 0:
            return np.mean(vectors, axis=0)
            
        weighted_avg = np.sum(vectors * weights, axis=0) / total_weight
        return weighted_avg

class SongRecommender:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.normpath(os.path.join(current_dir, '..', 'data'))
        
        # 1. Percorsi Nuovi (FAISS + SQLite)
        self.db_path = os.path.join(data_dir, 'tracks.db')
        self.index_path = os.path.join(data_dir, 'tracks.index')
        self.oracle_path = os.path.join(data_dir, 'oracle.pkl')
        self.blacklist_path = os.path.join(data_dir, 'blacklist.txt')

        # 2. Caricamento FAISS Index
        print(f"Recommender: Caricamento FAISS da {self.index_path}")
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            print(f"ERRORE CRITICO: Indice FAISS non trovato in {self.index_path}")
            self.index = None

        # 3. Caricamento Oracle
        if os.path.exists(self.oracle_path):
            try:
                self.oracle = joblib.load(self.oracle_path)
            except:
                self.oracle = None
        else:
            self.oracle = None
        
        self.audio_cols = ['energy', 'valence', 'danceability', 'tempo', 'loudness', 
                           'speechiness', 'acousticness', 'instrumentalness', 'liveness']

    def _get_current_context(self, user_history_df, manual_target=None):
        """
        Calcola il vettore target.
        Se c'è un target manuale (dagli slider), usa quello.
        Altrimenti usa la media ponderata della history.
        """
        # Se l'utente ha toccato gli slider, manual_target non è None
        if manual_target is not None:
            if isinstance(manual_target, dict):
                 return np.array([manual_target.get(col, 0.5) for col in self.audio_cols])
            return np.array(manual_target)

        # Altrimenti calcolo automatico
        if user_history_df.empty:
            return np.array([0.5]*9)
        
        # Calcolo Media Ponderata usando la colonna 'weight'
        if 'weight' in user_history_df.columns:
            return calculate_weighted_context(user_history_df, self.audio_cols)
        else:
            return user_history_df[self.audio_cols].mean().values

    def _get_candidates_from_faiss(self, query_vector, k=200):
        """Usa FAISS per trovare i candidati matematicamente più vicini."""
        if self.index is None: return pd.DataFrame()

        # FAISS richiede float32
        query_vector = query_vector.astype('float32').reshape(1, -1)
        
        # Ricerca (D=distanze, I=indici)
        distances, indices = self.index.search(query_vector, k)
        
        # Flattening degli array
        faiss_ids = indices[0]
        faiss_scores = 1 / (1 + distances[0]) # Convertiamo distanza L2 in similarità (0-1)
        
        # Recupero Metadati da SQLite
        conn = sqlite3.connect(self.db_path)
        
        # Query ottimizzata: WHERE faiss_id IN (...)
        ids_str = ','.join(map(str, faiss_ids))
        query = f"SELECT * FROM tracks WHERE faiss_id IN ({ids_str})"
        
        candidates = pd.read_sql_query(query, conn)
        conn.close()
        
        # Aggiungiamo lo score calcolato da FAISS al dataframe
        # Creiamo una mappa id -> score
        score_map = dict(zip(faiss_ids, faiss_scores))
        candidates['audio_score'] = candidates['faiss_id'].map(score_map)
        
        # Aggiungiamo anche le feature audio al dataframe dei candidati
        # Questo serve per il "Track DNA" nell'app
        try:
            vectors = self.index.reconstruct_batch(faiss_ids.astype('int64'))
            # Creiamo un DF temporaneo con le feature
            feats_df = pd.DataFrame(vectors, columns=self.audio_cols)
            feats_df['faiss_id'] = faiss_ids
            
            # Merge con i candidati
            candidates = pd.merge(candidates, feats_df, on='faiss_id')
        except Exception as e:
            print(f"Warning: Impossibile ricostruire vettori da FAISS ({e}). I dati audio saranno vuoti.")
        
        return candidates

    def recommend(self, user_history_df, k=20, target_features=None):
        """
        Genera raccomandazioni usando FAISS + Reranking.
        target_features: Dizionario opzionale con valori manuali (dagli slider).
        """
        if self.index is None:
            raise ValueError("Indice FAISS non caricato.")

        # 1. PREVISIONE VETTORE TARGET
        if target_features:
            # Modalità Manuale (Slider)
            predicted_vector = self._get_current_context(user_history_df, manual_target=target_features)
        elif hasattr(self, 'oracle') and self.oracle:
            # Modalità AI (Oracle)
            current_context = self._get_current_context(user_history_df)
            predicted_vector = self.oracle.predict_target(current_context).reshape(1, -1)
        else:
            # Fallback (Media Ponderata)
            predicted_vector = self._get_current_context(user_history_df).reshape(1, -1)

        # 2. RICERCA CANDIDATI (FAISS)
        # Cerchiamo più brani del necessario (es. 200) per poter filtrare dopo
        candidates = self._get_candidates_from_faiss(predicted_vector.flatten(), k=200)
        
        if candidates.empty:
            return pd.DataFrame(), predicted_vector

        # 3. FILTRAGGIO BLACKLIST & HISTORY
        exclude_ids = []
        if os.path.exists(self.blacklist_path):
            with open(self.blacklist_path, 'r') as f:
                exclude_ids.extend([line.strip() for line in f.readlines()])
        
        if not user_history_df.empty and 'id' in user_history_df.columns:
            exclude_ids.extend(user_history_df['id'].unique().tolist())
            
            # Recupero Top Artists per il bonus
            top_artists_set = set(user_history_df['artist'].unique())
            avg_pop = user_history_df['popularity'].mean() if 'popularity' in user_history_df else 50
            target_year = user_history_df['year'].mean() if 'year' in user_history_df.columns else 2020
        else:
            top_artists_set = set()
            avg_pop = 50
            target_year = 2020

        # Filtriamo via gli ID esclusi
        # Nota: nel DB SQLite la colonna è 'spotify_id', nella history è 'id'
        candidates = candidates[~candidates['spotify_id'].isin(exclude_ids)]

        # 4. CALCOLO SCORE SECONDARI (Reranking)
        # A. Temporal Score
        if 'year' in candidates.columns:
            candidates['year_diff'] = np.abs(candidates['year'] - target_year)
            candidates['year_score'] = 1 / (1 + (candidates['year_diff'] * 0.1))
        else:
            candidates['year_score'] = 0.5

        # B. Popularity Score
        if 'popularity' in candidates.columns:
            candidates['pop_diff'] = np.abs(candidates['popularity'] - avg_pop)
            candidates['pop_score'] = 1 / (1 + (candidates['pop_diff'] * 0.05))
        else:
            candidates['pop_score'] = 0.5
        
        # C. Artist Match
        candidates['is_top_artist'] = candidates['artist'].isin(top_artists_set)

        # 5. SCORE FINALE
        # Audio (FAISS) pesa il 60%, Anno 20%, Popolarità 20%
        candidates['final_score'] = (
            (candidates['audio_score'] * 0.60) + 
            (candidates['year_score'] * 0.20) + 
            (candidates['pop_score'] * 0.20)
        )
        
        # Bonus Artista (+30%)
        candidates.loc[candidates['is_top_artist'], 'final_score'] *= 1.30

        # 6. SELEZIONE FINALE
        sorted_candidates = candidates.sort_values(by='final_score', ascending=False)
        
        # Standardizzazione colonne per l'output (l'app si aspetta 'id', non 'spotify_id')
        sorted_candidates = sorted_candidates.rename(columns={'spotify_id': 'id'})
        
        # Selezione finale
        final_list = sorted_candidates.head(k).copy()
        
        # Aggiungiamo reason text
        def get_reason(row):
            if row['is_top_artist']: return "DNA Artista"
            if row['audio_score'] > 0.96: return "Vibe Identica"
            return "Scoperta AI"

        final_list['reason_text'] = final_list.apply(get_reason, axis=1)
        final_list['match_percentage'] = final_list['final_score'] * 100
        
        return final_list, predicted_vector