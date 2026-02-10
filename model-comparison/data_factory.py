import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os

# Vengono ignorati genere e popolarità, usiamo solo feature audio numeriche
AUDIO_FEATURES = [
    'energy', 'valence', 'danceability', 'tempo', 'loudness', 
    'speechiness', 'acousticness', 'instrumentalness', 'liveness'
]

class MusicSequenceDataset(Dataset):
    def __init__(self, data_matrix, seq_length=20):
        """
        Dataset con logica 'Growing Window' (Finestra Crescente) + Padding.
        
        Obiettivo: Sfruttare TUTTI i dati fin dalla seconda canzone.
        - Se ho solo la canzone 1 -> Input: [0,0..., Canzone1] -> Target: Canzone2
        - Se ho Canzone 1,2 -> Input: [0,0..., Canzone1, Canzone2] -> Target: Canzone3
        
        Args:
            data_matrix (np.array): Matrice delle feature [N_samples, N_features]
            seq_length (int): Lunghezza MASSIMA della finestra temporale (per il padding)
        """
        self.seq_length = seq_length
        # Trasformiamo i dati numpy in Tensori PyTorch (standard float32)
        self.data = torch.tensor(data_matrix, dtype=torch.float32)
        self.n_features = self.data.shape[1]
    
    def __len__(self):
        # Con la Growing Window, possiamo predire dalla seconda canzone in poi.
        return max(0, len(self.data) - 1)
    
    def __getitem__(self, idx):
        """
        idx indica l'indice della canzone che vogliamo PREDIRE (il target).
        Poiché idx parte da 0, usiamo 'idx + 1' come indice reale nel dataset.
        """
        # Indice della canzone target (quella da predire)
        target_idx = idx + 1
        
        # Recuperiamo il Target
        y = self.data[target_idx]
        
        # Costruiamo l'Input (lo storico precedente)
        # Prendiamo tutto ciò che c'è prima del target, fino a un massimo di 'seq_length' indietro
        start_idx = max(0, target_idx - self.seq_length)
        sequence = self.data[start_idx : target_idx]
        
        # --- PADDING ---
        # Se la sequenza è più corta di seq_length, aggiungiamo zeri all'inizio (Padding a sinistra)
        curr_len = sequence.shape[0]
        if curr_len < self.seq_length:
            pad_len = self.seq_length - curr_len
            # Creiamo un tensore di zeri [pad_len, n_features]
            padding = torch.zeros((pad_len, self.n_features), dtype=torch.float32)
            # Concateniamo: Zeri + Sequenza Reale
            x = torch.cat([padding, sequence], dim=0)
        else:
            x = sequence
            
        return x, y

def create_dataloaders(csv_path, seq_length=20, batch_size=32, test_split=0.2, max_rows=None):
    """
    Caricamento dati, pulizia e creazione Dataset per training e test.
    
    NOTA: Restituisce oggetti Dataset (non DataLoader). 
    I DataLoader vanno creati nel main script per gestire correttamente i seed.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File non trovato: {csv_path}")
    
    print(f"[INFO] Database factory, caricamento: {csv_path}...")
    df = pd.read_csv(csv_path)

    # Limitazione righe (opzionale, per test rapidi)
    if max_rows is not None:
        df = df.tail(max_rows) # Prendi le ultime n righe (le più recenti)
        print(f"[INFO] Limitato a ultime {max_rows} canzoni")

    # Ordinamento temporale (Cruciale: passato -> futuro)
    if 'played_at' in df.columns:
        df['played_at'] = pd.to_datetime(df['played_at'], format='mixed')
        df = df.sort_values('played_at').reset_index(drop=True)
    else:
        print("[WARN] Colonna 'played_at' assente. Si assume che l'ordine nel CSV sia corretto.")

    # Controllo colonne mancanti
    missing_cols = [c for c in AUDIO_FEATURES if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonne mancanti: {missing_cols}")
    
    # Creazione matrice numpy (float32)
    data_matrix = df[AUDIO_FEATURES].values.astype(np.float32)
    total_samples = len(data_matrix)

    # --- SPLIT CRONOLOGICO PURO (CLEAN SPLIT) ---
    # Train: primi 80% dei dati
    # Test: ultimi 20% dei dati
    # Nessun overlap per garantire la validità scientifica del test.
    
    test_size = int(total_samples * test_split)
    train_size = total_samples - test_size

    # Split dei dati grezzi
    train_data = data_matrix[:train_size]
    test_data = data_matrix[train_size:] 
    
    # Controllo dimensioni minime
    if len(train_data) < 2 or len(test_data) < 2:
        print(f"[WARN] Dataset troppo piccolo dopo lo split. Train: {len(train_data)}, Test: {len(test_data)}")

    # Creazione Dataset
    train_dataset = MusicSequenceDataset(train_data, seq_length)
    test_dataset = MusicSequenceDataset(test_data, seq_length)

    print(f"[INFO] Dataset creati (Split Sequenziale):")
    print(f" - Train Samples: {len(train_dataset)} (da {len(train_data)} canzoni raw)")
    print(f" - Test Samples:  {len(test_dataset)} (da {len(test_data)} canzoni raw)")

    # Restituisce i dataset e il numero di feature
    return train_dataset, test_dataset, len(AUDIO_FEATURES)

if __name__ == "__main__":
    # Test rapido se eseguiamo direttamente questo file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Aggiusta il percorso se necessario, qui assume struttura standard
    data_path = os.path.join(base_dir, '..', 'data', 'user_history.csv')

    try:
        # Test con una finestra piccola per vedere il padding in azione
        tr_ds, te_ds, n = create_dataloaders(data_path, seq_length=5, batch_size=2, max_rows=20)
        
        print("\n--- TEST DATASET (TRAIN) ---")
        if len(tr_ds) > 0:
            x, y = tr_ds[0]
            print(f"Shape Input (X): {x.shape} -> [Seq_Len, Features]")
            print(f"Shape Target (y): {y.shape} -> [Features]")
        else:
            print("Dataset Train vuoto.")
        
        print("\n--- TEST DATASET (TEST) ---")
        if len(te_ds) > 0:
            x_test, y_test = te_ds[0]
            print(f"Test Input Shape: {x_test.shape}")
            # Verifica Padding: Poiché lo split è pulito, il primo elemento del test 
            # non ha storia pregressa nel vettore test_data, quindi sarà quasi tutto padding (zeri).
            print("Primo input del Test Set (Dovrebbe mostrare padding/zeri iniziali):")
            print(x_test)
        else:
            print("Dataset Test vuoto.")

    except Exception as e:
        print(f"Errore o file non trovato per il test rapido: {e}")