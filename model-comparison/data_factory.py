import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import os

#vengono ignorati genere e popolarità
AUDIO_FEATURES = [
    'energy', 'valence', 'danceability', 'tempo', 'loudness', 
    'speechiness', 'acousticness', 'instrumentalness', 'liveness'
]

class MusicSequenceDataset(Dataset):
    def __init__(self, data_matrix, seq_length=20):
        
         #creazione dataset Pytorch per sequenze musicali.
         
         #Trasforma una matrice di dati [N_samples, N_features] in sequenze per addestramento
        self.seq_length = seq_length

        #trasformiamo i dati numpy in Tensory PyTorch (standard float32)
        self.data = torch.tensor(data_matrix, dtype=torch.float32)
    
    def __len__(self):
        #Con 100 canzoni, organizzate in finestra da 20 ---> ho 80 sequenze possibili
        #seq 1: 0-19 -> target 20
        #seq 1: 1-20 -> target 21
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        # INPUT (X): Finestra di 'seq_length' canzoni (nel nostro caso seq_length = 20)
        # Shape: [seq_length, n_features]
        x = self.data[idx : idx + self.seq_length]
        
        # TARGET (y): La canzone esattamente successiva alla finestra
        # Shape: [n_features]
        y = self.data[idx + self.seq_length]
        
        return x, y

def create_dataloaders(csv_path, seq_length=20, batch_size=32, test_split=0.2, max_rows = None):

    #caricamento dei dati, pulizia e creazione data loader per il training
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File non trovato: {csv_path}")
    
    print(f"Database factory, caricamento:{csv_path}...")
    df = pd.read_csv(csv_path)


    if max_rows is not None:
        original_len = len(df)
        df = df.tail(max_rows) #prendi le ultime n righe, le più recenti
        print(f"Limitato a ultime {max_rows} canzoni")

    #ordinamento temporale (importante in modo da non predire il passato con il futuro)
    if 'played_at' in df.columns:
        df['played_at'] = pd.to_datetime(df['played_at'], format='mixed')
        df = df.sort_values('played_at').reset_index(drop=True)
    else:
        print("Errore, colonne played_at assente. ")

    
    #selezione feature
    missing_cols = [c for c in AUDIO_FEATURES if c not in df.columns]

    if missing_cols:
        raise ValueError(f"Colonne mancanti: {missing_cols}")
    
    #Creazione matrice di soli numeri
    data_matrix = df[AUDIO_FEATURES].values.astype(np.float32)
    total_samples = len(data_matrix)

    #SPLIT cronologico (train = passato , test = futuro)
    test_size = int(total_samples * test_split)
    train_size = total_samples - test_size

    train_data = data_matrix[:train_size]

    #per fare il test, includiamo un buffer del train per avere lo storico della prima canzone
    buffer_start = train_size - seq_length
    if buffer_start < 0: buffer_start = 0
    test_data = data_matrix[buffer_start:]

    #Creazione dataset
    train_dataset = MusicSequenceDataset(train_data, seq_length)
    test_dataset = MusicSequenceDataset(test_data, seq_length)

    #dataloaders
    #drop_last = true --> scarta ultimo batch se incompleto
    #avendo solo 50 canzoni in un caso, andiamo a mettere false (in modo che creerà un bacth da 20 e addestramento verrà portato avanti)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Sequenze (Window={seq_length}): Train={len(train_dataset)}, Test={len(test_dataset)}")

    return train_loader, test_loader, len(AUDIO_FEATURES)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'data', 'user_history.csv')

    try:
        tr, te, n = create_dataloaders(data_path, seq_length=5, batch_size=4)
        x, y = next(iter(tr))

        print(f"Shape Input (X): {x.shape}")
        print(f"Shape Target (y): {y.shape}")

    except Exception as e:
        print(f"Errore: {e}")
