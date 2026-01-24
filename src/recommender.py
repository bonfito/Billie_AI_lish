import pandas as pd
import numpy as np
import os
import joblib
import re
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import calculate_avalanche_context

class SongRecommender:
    def __init__(self):
        # Inizializzazione percorsi
        self.tracks_path = os.path.join("data", "tracks_processed.csv")
        self.oracle_path = os.path.join("data", "oracle.pkl")

        # Caricamento DB Musicale
        if os.path.exists(self.tracks_path):
            self.df_tracks = pd.read_csv(self.tracks_path, low_memory=False)
        else:
            raise FileNotFoundError(f"Database non trovato in {self.tracks_path}")
        
        # Caricamento Modello AI
        if os.path.exists(self.oracle_path):
            self.oracle = joblib.load(self.oracle_path)
        else:
            raise FileNotFoundError(f"Modello Oracle non trovato in {self.oracle_path}")
        
        # Definizione features audio
        self.audio_cols = ['energy', 'valence', 'danceability', 'tempo', 'loudness', 
                           'speechiness', 'acousticness', 'instrumentalness', 'liveness']
        
        # Mappa Tonalità
        self.key_map = {
            0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F', 
            6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
        }

    def _get_current_context(self, user_history_df):
        current_context = user_history_df.loc[0, self.audio_cols].values
        for i in range(1, len(user_history_df)):
            track_data = user_history_df.loc[i, self.audio_cols].values
            current_context = calculate_avalanche_context(current_context, track_data, n=5)
        return current_context
    
    def recommend(self, user_history_df, k=20):
        # 1. PREVISIONE AI
        current_context = self._get_current_context(user_history_df)
        predicted_vector = self.oracle.predict_target(current_context).reshape(1, -1)
        noise = np.random.normal(0, 0.05, predicted_vector.shape)
        predicted_vector = predicted_vector + noise
    
        # 2. ANALISI UTENTE
        avg_popularity = user_history_df['popularity'].mean()
        top_artists = user_history_df['artist'].value_counts().head(10).index.tolist()
        
        target_year = 1990
        if 'year' in user_history_df.columns:
            target_year = user_history_df['year'].mean()
            
        print("\n" + "="*100)
        print(f"ANALISI TARGET AI:")
        print(f"Target Anno: {int(target_year)}")
        target_vals = [f"{col[:3]}:{val:.2f}" for col, val in zip(self.audio_cols, predicted_vector[0])]
        print(" | ".join(target_vals))
        print("="*100 + "\n")

        # 3. FILTRAGGIO BASE
        candidates = self.df_tracks.copy()
        candidates = candidates[candidates['speechiness'] < 0.6]

        if avg_popularity > 40:
            candidates = candidates[candidates['popularity'] >= 15]

        history_ids = user_history_df['id'].unique().tolist()
        candidates = candidates[~candidates['id'].isin(history_ids)]

        # 4. CALCOLO SCORE
        candidate_vectors = candidates[self.audio_cols].fillna(0.5)
        candidates['audio_score'] = cosine_similarity(predicted_vector, candidate_vectors)[0]
        
        candidates['year'] = candidates['year'].fillna(target_year)
        candidates['year_diff'] = np.abs(candidates['year'] - target_year)
        candidates['year_score'] = 1 / (1 + (candidates['year_diff'] * 0.1))
        
        candidates['is_top_artist'] = False
        if top_artists:
            safe_artists = [re.escape(str(a)) for a in top_artists]
            pattern = '|'.join(safe_artists)
            candidates['is_top_artist'] = candidates['artist'].astype(str).str.contains(pattern, case=False, na=False)
            candidates.loc[candidates['is_top_artist'], 'year_score'] = 1.0

        candidates['final_score'] = (candidates['audio_score'] * 0.70) + (candidates['year_score'] * 0.30)
        
        if top_artists:
            candidates.loc[candidates['is_top_artist'], 'final_score'] *= 1.10

        # --- FASE 5: SELEZIONE E DE-DUPLICAZIONE ---
        
        def get_reason(row):
            reasons = []
            if row['is_top_artist']: reasons.append("Top Artista")
            if row['audio_score'] > 0.95: reasons.append("Audio Perfetto")
            elif row['audio_score'] > 0.85: reasons.append("Vibe Simile")
            if row['year_diff'] < 3: reasons.append(f"Anno {int(row['year'])}")
            elif row['is_top_artist'] and row['year_diff'] > 10: reasons.append("Legacy")
            elif row['year_diff'] < 8: reasons.append("Epoca Vicina")
            if row['popularity'] > 75: reasons.append("Hit")
            return " | ".join(reasons[:2])

        # A. SELEZIONE STANDARD (con De-Duplicazione e Capping Artista)
        sorted_candidates = candidates.sort_values(by='final_score', ascending=False)
        
        # Filtro Anti-Doppioni (chiave unica Titolo-Artista)
        sorted_candidates['unique_key'] = (sorted_candidates['name'].str.lower().str.split(' - ').str[0] + 
                                           sorted_candidates['artist'].str.lower())
        sorted_candidates = sorted_candidates.drop_duplicates(subset='unique_key', keep='first')
        
        # --- MODIFICA: LOOP DI SELEZIONE CON ARTIST CAPPING ---
        standard_list = []
        artist_counts = {} # Dizionario per contare quante volte appare un artista
        
        for idx, row in sorted_candidates.iterrows():
            # Pulizia nome artista per il conteggio
            artist_name = row['artist'].strip()
            
            # Se l'artista è già apparso 2 volte, lo saltiamo per dare spazio ad altri
            if artist_counts.get(artist_name, 0) >= 2:
                continue
                
            # Aggiungiamo alla lista
            row['reason_text'] = get_reason(row)
            row['match_percentage'] = np.minimum(row['final_score'] * 100, 99.9)
            standard_list.append(row)
            
            # Aggiorniamo il contatore
            artist_counts[artist_name] = artist_counts.get(artist_name, 0) + 1
            
            # Ci fermiamo quando ne abbiamo 15
            if len(standard_list) >= (k - 5):
                break
        
        standard_recs = pd.DataFrame(standard_list)
        # --------------------------------------------------------

        # B. SELEZIONE WILDCARD
        exclude_ids = standard_recs['id'].tolist()
        wild_candidates = candidates[
            (~candidates['id'].isin(exclude_ids)) & 
            (candidates['popularity'] > 60) & 
            (candidates['audio_score'] < 0.6) & 
            (candidates['audio_score'] > 0.3)
        ]
        
        wild_candidates['unique_key'] = (wild_candidates['name'].str.lower().str.split(' - ').str[0] + 
                                         wild_candidates['artist'].str.lower())
        wild_candidates = wild_candidates.drop_duplicates(subset='unique_key', keep='first')

        if len(wild_candidates) < 5:
             wild_candidates = candidates[
                (~candidates['id'].isin(exclude_ids)) & 
                (candidates['popularity'] > 50)
            ].drop_duplicates(subset=['name', 'artist'])

        wildcards = pd.DataFrame()
        if not wild_candidates.empty:
            sample_size = min(5, len(wild_candidates))
            wildcards = wild_candidates.sample(n=sample_size).copy()
            wildcards['reason_text'] = "Wildcard (Sorpresa)"
            wildcards['match_percentage'] = wildcards['audio_score'] * 100 

        # C. UNIONE
        final_recs = pd.concat([standard_recs, wildcards])
        
        return final_recs, predicted_vector


if __name__ == "__main__":
    try:
        # Carichamento
        history_path = os.path.join("data", "user_history.csv")
        if not os.path.exists(history_path):
            print("User history non trovata.")
            exit()
                
        df_user = pd.read_csv(history_path)
        engine = SongRecommender()
            
        print("Generazione raccomandazioni (20 Uniche)...")
        recs, _ = engine.recommend(df_user, k=20)
            
        print("\nTOP 20 SUGGERIMENTI:")
        print("=" * 120)
        
        abbr = {
            'energy': 'En', 'valence': 'Va', 'danceability': 'Da', 'tempo': 'Te', 
            'loudness': 'Lo', 'speechiness': 'Sp', 'acousticness': 'Ac', 
            'instrumentalness': 'In', 'liveness': 'Li'
        }
            
        for i, (idx, row) in enumerate(recs.iterrows(), 1):
            title = (str(row['name'])[:28] + '..') if len(str(row['name'])) > 28 else str(row['name'])
            artist = str(row['artist']).replace("['", "").replace("']", "").replace("'", "")[:18]
            year = int(row['year']) if pd.notna(row['year']) else 0
            
            # Key e Mode
            key_val = int(row['key']) if pd.notna(row['key']) and row['key'] != -1 else -1
            mode_val = int(row['mode']) if pd.notna(row['mode']) else -1
            
            key_str = engine.key_map.get(key_val, "?")
            mode_str = "Min" if mode_val == 0 else "Maj" if mode_val == 1 else ""
            tonality = f"{key_str} {mode_str}" if key_str != "?" else "N/A"

            # Vettore
            feat_parts = []
            for col in engine.audio_cols:
                val = row[col]
                label = abbr.get(col, col[:2])
                feat_parts.append(f"{label}:{val:.2f}")
            feat_str = " ".join(feat_parts)
            
            if i == 16:
                print("-" * 120)
                print("   --- ZONA WILDCARD (Sorprese) ---")
                print("-" * 120)

            print(f"{i:<2} {title:<30} {artist:<20} {year:<5} {tonality:<8} {row['match_percentage']:.1f}%")
            print(f"   > {row['reason_text']}")
            print(f"   > Vettore: {feat_str}")
            print("-" * 120)
                
    except Exception as e:
        print(f"Errore: {e}")