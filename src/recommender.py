import pandas as pd
import numpy as np
import os
import joblib
import sqlite3
import faiss

class SongRecommender:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.normpath(os.path.join(current_dir, '..', 'data'))
        
        self.db_path = os.path.join(data_dir, 'tracks.db')
        self.index_path = os.path.join(data_dir, 'tracks.index')
        self.oracle_path = os.path.join(data_dir, 'oracle.pkl')
        self.blacklist_path = os.path.join(data_dir, 'blacklist.txt')

        # Caricamento FAISS Index
        print(f"Recommender: Caricamento FAISS da {self.index_path}")
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            print(f"ERRORE CRITICO: Indice FAISS non trovato in {self.index_path}")
            self.index = None

        # Caricamento Oracle (opzionale)
        if os.path.exists(self.oracle_path):
            try:
                self.oracle = joblib.load(self.oracle_path)
            except:
                self.oracle = None
        else:
            self.oracle = None
        
        self.audio_cols = ['energy', 'valence', 'danceability', 'tempo', 'loudness', 
                           'speechiness', 'acousticness', 'instrumentalness', 'liveness']

    def _get_current_context(self, history_df, manual_target=None):
        """
        Calcola il vettore target (Media Semplice).
        """
        # 1. Slider Manuali (Vincono su tutto)
        if manual_target is not None:
             # Assicuriamoci che sia un array numpy
             if isinstance(manual_target, dict):
                 return np.array([manual_target.get(col, 0.5) for col in self.audio_cols])
             return np.array(manual_target)

        if history_df.empty:
            return np.array([0.5]*9)

        # 2. Focus Sessione (Ultimi 10 brani ascoltati)
        # Assumiamo che history_df sia già ordinato correttamente da fetch_userhistory
        # (ovvero i più recenti in alto, riga 0, 1, 2...)
        # Quindi prendiamo semplicemente la testa del dataframe.
        
        recent_session = history_df.head(10)

        # Calcolo Media Semplice (Logica Pura)
        # Se un utente ascolta Trap e poi Jazz, la media si sposterà velocemente.
        session_vector = recent_session[self.audio_cols].mean(numeric_only=True).values
        
        # print(f"🎯 Target basato sulla media degli ultimi {len(recent_session)} brani.")
        return session_vector

    def _get_candidates_from_faiss(self, query_vector, k=200):
        if self.index is None: return pd.DataFrame()

        query_vector = query_vector.astype('float32').reshape(1, -1)
        # Ricerca vettoriale pura
        distances, indices = self.index.search(query_vector, k)
        
        faiss_ids = indices[0]
        # Score di similarità (più vicino = score più alto)
        # Usiamo 1 / (1 + distanza) per avere un valore tra 0 e 1
        faiss_scores = 1 / (1 + distances[0]) 
        
        # Recupero info da SQLite
        conn = sqlite3.connect(self.db_path)
        ids_str = ','.join(map(str, faiss_ids))
        query = f"SELECT * FROM tracks WHERE faiss_id IN ({ids_str})"
        candidates = pd.read_sql_query(query, conn)
        conn.close()
        
        # Mappa score
        score_map = dict(zip(faiss_ids, faiss_scores))
        candidates['audio_score'] = candidates['faiss_id'].map(score_map)
        
        # Recupero vettori audio per l'interfaccia (opzionale ma utile per il radar)
        try:
            vectors = self.index.reconstruct_batch(faiss_ids.astype('int64'))
            feats_df = pd.DataFrame(vectors, columns=self.audio_cols)
            feats_df['faiss_id'] = faiss_ids
            candidates = pd.merge(candidates, feats_df, on='faiss_id')
        except Exception:
            pass
        
        return candidates

    def recommend(self, user_history_df, k=20, target_features=None):
        if self.index is None:
            raise ValueError("Indice FAISS non caricato.")

        # 1. Calcolo Vettore Target (Puro e Semplice)
        predicted_vector = self._get_current_context(user_history_df, manual_target=target_features)

        # 2. Ricerca FAISS
        candidates = self._get_candidates_from_faiss(predicted_vector.flatten(), k=200)
        
        if candidates.empty:
            return pd.DataFrame(), predicted_vector

        # 3. Filtro Blacklist / Già ascoltati
        exclude_ids = []
        if os.path.exists(self.blacklist_path):
            with open(self.blacklist_path, 'r') as f:
                exclude_ids.extend([line.strip() for line in f.readlines()])
        
        if not user_history_df.empty and 'id' in user_history_df.columns:
            exclude_ids.extend(user_history_df['id'].unique().tolist())

        # Filtra
        # Nota: nel DB SQLite la colonna è 'spotify_id'
        candidates = candidates[~candidates['spotify_id'].isin(exclude_ids)]

        # 4. Ranking Semplice (Audio è Re)
        # Nessun bonus artista, nessuna penalità anno/popolarità. 
        # Chi è matematicamente più simile vince.
        candidates['final_score'] = candidates['audio_score']
        
        # 5. Selezione Finale
        sorted_candidates = candidates.sort_values(by='final_score', ascending=False)
        sorted_candidates = sorted_candidates.rename(columns={'spotify_id': 'id'})
        
        final_list = sorted_candidates.head(k).copy()
        
        # Testi descrittivi semplici
        def get_reason(row):
            if row['audio_score'] > 0.95: return "Match Perfetto"
            if row['audio_score'] > 0.85: return "Alta Affinità"
            return "Consigliato"

        final_list['reason_text'] = final_list.apply(get_reason, axis=1)
        final_list['match_percentage'] = final_list['final_score'] * 100
        
        return final_list, predicted_vector