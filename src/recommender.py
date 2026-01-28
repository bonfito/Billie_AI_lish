import pandas as pd
import numpy as np
import os
import joblib
import faiss  # <--- Libreria essenziale per la velocità
from sklearn.preprocessing import normalize

# --- IMPORT INTELLIGENTE ---
try:
    from utils import calculate_avalanche_context
except ImportError:
    from src.utils import calculate_avalanche_context

class SongRecommender:
    def __init__(self, dataset_path=None):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.normpath(os.path.join(current_dir, '..', 'data'))
        
        # 1. Configurazione Percorsi
        self.tracks_path = dataset_path if dataset_path else os.path.join(data_dir, 'tracks_processed.csv')
        self.index_path = os.path.join(data_dir, 'tracks.index')
        self.oracle_path = os.path.join(data_dir, 'oracle.pkl')
        self.blacklist_path = os.path.join(data_dir, 'blacklist.txt')

        self.audio_cols = ['energy', 'valence', 'danceability', 'tempo', 'loudness', 
                           'speechiness', 'acousticness', 'instrumentalness', 'liveness']

        # 2. Caricamento Dataframe
        print(f"✅ Recommender: Caricamento DB da {self.tracks_path}")
        if os.path.exists(self.tracks_path):
            # Supporto sia CSV che Parquet
            if self.tracks_path.endswith('.parquet'):
                self.df_tracks = pd.read_parquet(self.tracks_path)
            else:
                self.df_tracks = pd.read_csv(self.tracks_path, low_memory=False)
            
            # Reset index per garantire allineamento con FAISS (0..N)
            self.df_tracks = self.df_tracks.reset_index(drop=True)
        else:
            print(f"❌ DATABASE NON TROVATO: {self.tracks_path}")
            self.df_tracks = pd.DataFrame()

        # 3. Gestione Indice FAISS
        if not self.df_tracks.empty:
            if os.path.exists(self.index_path):
                print(f"🔹 Caricamento Indice FAISS da {self.index_path}")
                try:
                    self.index = faiss.read_index(self.index_path)
                except:
                    print("⚠️ Indice corrotto. Rigenerazione...")
                    self.build_index()
            else:
                print("⚙️ Creazione nuovo Indice FAISS...")
                self.build_index()
        else:
            self.index = None

        # 4. Caricamento Oracle
        if os.path.exists(self.oracle_path):
            try:
                self.oracle = joblib.load(self.oracle_path)
            except:
                self.oracle = None
        else:
            self.oracle = None

    def build_index(self):
        """Costruisce indice FAISS basato su Inner Product (Cosine Similarity)."""
        # Estrai matrice feature
        vectors = self.df_tracks[self.audio_cols].fillna(0.5).values.astype('float32')
        
        # Normalizza vettori (L2) per fare in modo che Inner Product = Cosine Similarity
        faiss.normalize_L2(vectors)
        
        # Dimensione vettori
        d = vectors.shape[1]
        
        # IndexFlatIP = Exact Search con Inner Product
        self.index = faiss.IndexFlatIP(d)
        self.index.add(vectors)
        
        faiss.write_index(self.index, self.index_path)
        print(f"✅ Indice FAISS salvato ({self.index.ntotal} vettori).")

    def _get_current_context(self, user_history_df):
        """Calcola il vettore medio pesato (Avalanche)."""
        if user_history_df.empty:
            return np.array([0.5]*9)
            
        current_context = user_history_df.iloc[0][self.audio_cols].values
        for i in range(1, len(user_history_df)):
            track_data = user_history_df.iloc[i][self.audio_cols].values
            current_context = calculate_avalanche_context(current_context, track_data, n=5)
        return current_context
    
    def recommend(self, user_history_df, k=20, target_features=None):
        """
        Genera raccomandazioni usando FAISS (Velocissimo) + Filtri Pandas.
        Accetta 'target_features' opzionale per controllo manuale (slider).
        """
        if self.index is None or self.df_tracks.empty:
            return pd.DataFrame(), np.zeros(9)

        # 1. DETERMINA VETTORE TARGET 
        if target_features is not None:
            # Se l'utente usa gli slider, usiamo quelli direttamente
            if isinstance(target_features, dict):
                predicted_vector = np.array([target_features.get(c, 0.5) for c in self.audio_cols])
            else:
                predicted_vector = np.array(target_features)
            predicted_vector = predicted_vector.reshape(1, -1)
        
        elif hasattr(self, 'oracle') and self.oracle:
            # Altrimenti usa l'AI
            current_context = self._get_current_context(user_history_df)
            predicted_vector = self.oracle.predict_target(current_context).reshape(1, -1)
        
        else:
            # Fallback media
            valid_cols = [c for c in self.audio_cols if c in user_history_df.columns]
            if valid_cols:
                predicted_vector = user_history_df[valid_cols].mean().values.reshape(1, -1)
            else:
                predicted_vector = np.array([0.5]*9).reshape(1, -1)

        # Aggiungi rumore per evitare stagnazione
        noise = np.random.normal(0, 0.02, predicted_vector.shape)
        search_vector = np.clip(predicted_vector + noise, 0, 1).astype('float32')
        
        # Normalizza per FAISS (per Cosine Similarity)
        faiss.normalize_L2(search_vector)

        # 2. RICERCA FAISS (Core veloce)
        # Cerchiamo K * 50 candidati per avere margine dopo i filtri (blacklist, artista, ecc.)
        k_search = min(1000, len(self.df_tracks))
        distances, indices = self.index.search(search_vector, k_search)
        
        # Recupera le righe dal DataFrame originale usando gli indici restituiti da FAISS
        # indices[0] contiene gli ID riga.
        candidate_indices = indices[0]
        # Filtra eventuali -1 (se il DB è piccolo)
        candidate_indices = candidate_indices[candidate_indices != -1]
        
        candidates = self.df_tracks.iloc[candidate_indices].copy()
        # Assegna lo score (che qui è il Cosine Similarity già calcolato da FAISS)
        # distances[0] corrisponde ai candidati in ordine
        candidates['audio_score'] = distances[0][:len(candidates)]

        # 3. FILTRAGGIO (Blacklist e History)
        exclude_ids = []
        if os.path.exists(self.blacklist_path):
            with open(self.blacklist_path, 'r') as f:
                exclude_ids.extend([line.strip() for line in f.readlines()])
        
        if 'id' in user_history_df.columns:
            exclude_ids.extend(user_history_df['id'].unique().tolist())
        
        candidates = candidates[~candidates['id'].isin(exclude_ids)]

        # 4. CALCOLO SCORE SECONDARI (Anno, Popolarità, Artista)
        target_year = user_history_df['year'].mean() if not user_history_df.empty and 'year' in user_history_df.columns else 2022
        avg_pop = user_history_df['popularity'].mean() if not user_history_df.empty and 'popularity' in user_history_df else 50
        top_artists = set(user_history_df['artist'].unique()) if not user_history_df.empty and 'artist' in user_history_df else set()

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
        if top_artists and 'artist' in candidates.columns:
            candidates['is_top_artist'] = candidates['artist'].isin(top_artists)

        # 5. SCORE FINALE
        candidates['final_score'] = (
            (candidates['audio_score'] * 0.65) + 
            (candidates['year_score'] * 0.15) + 
            (candidates['pop_score'] * 0.20)
        )
        
        if top_artists:
            candidates.loc[candidates['is_top_artist'], 'final_score'] *= 1.25

        # 6. SELEZIONE FINALE
        def get_reason(row):
            if row['is_top_artist']: return "DNA Artista"
            if row['audio_score'] > 0.95: return "Vibe Identica"
            if row['pop_score'] > 0.90: return "Hit Affine"
            return "Consigliato"

        # Ordina e deduplica (Case Sensitive Key)
        candidates = candidates.sort_values(by='final_score', ascending=False)
        if 'name' in candidates.columns and 'artist' in candidates.columns:
            candidates['unique_key'] = candidates['name'].astype(str) + candidates['artist'].astype(str)
            candidates = candidates.drop_duplicates(subset='unique_key', keep='first')

        # --- A. STANDARD LIST ---
        standard_list = []
        artist_counts = {}
        for _, row in candidates.iterrows():
            if len(standard_list) >= k: break
            
            # Limite artista
            a_name = str(row.get('artist', ''))
            if artist_counts.get(a_name, 0) >= 2: continue
            
            row['reason_text'] = get_reason(row)
            row['match_percentage'] = min(row['final_score'] * 100, 99)
            standard_list.append(row)
            artist_counts[a_name] = artist_counts.get(a_name, 0) + 1
            
        final_df = pd.DataFrame(standard_list)

        # --- B. WILDCARD VELOCI ---
        # Per le wildcard non scansioniamo tutto il DB (lento), ma campioniamo brani popolari
        # e calcoliamo la distanza solo su quelli.
        if len(final_df) < k + 5:
            # Prendi brani ad alta popolarità dal DB originale che NON sono già candidati
            high_pop = self.df_tracks[self.df_tracks['popularity'] > 60]
            if not high_pop.empty:
                # Campiona 500 brani a caso per essere veloci
                sample_wild = high_pop.sample(n=min(500, len(high_pop)))
                # Escludi quelli già visti
                current_ids = final_df['id'].tolist() if not final_df.empty else []
                sample_wild = sample_wild[~sample_wild['id'].isin(current_ids + exclude_ids)]
                
                if not sample_wild.empty:
                    # Calcola similarità vettoriale solo su questo piccolo campione
                    w_vectors = sample_wild[self.audio_cols].fillna(0.5).values.astype('float32')
                    faiss.normalize_L2(w_vectors)
                    
                    # Prodotto scalare manuale (veloce su 500 righe)
                    # search_vector è già normalizzato (1, D)
                    # w_vectors è (500, D)
                    # Risultato (1, 500)
                    scores = np.dot(search_vector, w_vectors.T).flatten()
                    
                    sample_wild['audio_score'] = scores
                    
                    # Filtra quelli "diversi" (similarità tra 0.4 e 0.7)
                    valid_wild = sample_wild[sample_wild['audio_score'].between(0.3, 0.7)]
                    
                    if not valid_wild.empty:
                        picked = valid_wild.head(5).copy()
                        picked['reason_text'] = "Wildcard (Novità)"
                        picked['match_percentage'] = picked['audio_score'] * 100
                        final_df = pd.concat([final_df, picked], ignore_index=True)

        # Shuffle finale
        if not final_df.empty:
            final_df = final_df.sample(frac=1).reset_index(drop=True)

        return final_df, predicted_vector.flatten()