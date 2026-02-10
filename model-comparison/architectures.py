import torch
import torch.nn as nn
import math

# ==============================================================================
# Primo Modello - MLP (Feed Forward)
# ==============================================================================
class BillieMLP(nn.Module):
    """
    - Appiattisce la sequenza in un vettore unico.
    - Non capisce l'ordine temporale (tratta la sequenza come un'immagine statica).
    
    ARCHITETTURA:
    INPUT(20,9) --> Flatten --> Dense(256) --> ReLU --> Dropout --> Dense(128) --> ReLU --> Output(9)
    """

    def __init__(self, seq_length=20, input_size=9, hidden_size=256, dropout=0.1):
        super(BillieMLP, self).__init__()

        # Dimensione dopo il flatten: 20 * 9 = 180
        self.flattened_size = seq_length * input_size 

        # Layer densi (Fully Connected)
        self.fc1 = nn.Linear(self.flattened_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, input_size)

        # Attivazione e Regolarizzazione
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) # Aggiunto per evitare overfitting su dataset piccoli

    def forward(self, x):
        """
        Input: (batch, seq_len, features) -> (32, 20, 9)
        Output: (batch, features) -> (32, 9)
        """
        batch_size = x.size(0)
        
        # Step 1: Flatten (Appiattimento)
        # Da (32, 20, 9) a (32, 180)
        x = x.view(batch_size, -1) 

        # Step 2: Passaggio nei layer con Dropout
        x = self.relu(self.fc1(x))
        x = self.dropout(x)          # Dropout applicato dopo la prima attivazione
        x = self.relu(self.fc2(x))
        x = self.fc3(x)              # Nessuna attivazione finale (Regressione pura)

        return x


# ==============================================================================
# Secondo Modello - LSTM (Recurrent)
# ==============================================================================
class BillieLSTM(nn.Module):
    """
    Long Short-Term Memory
    
    - Legge le canzoni sequenzialmente.
    - Mantiene una memoria interna che si aggiorna ad ogni timestamp.
    - Ideale per dataset sequenziali medio-piccoli.
    
    ARCHITETTURA:
    Input(20,9) -> LSTM(128, 2 layers) -> Ultimo Hidden State -> Dense(9)
    """

    def __init__(self, input_size=9, hidden_size=128, num_layers=2, dropout=0.1):
        super(BillieLSTM, self).__init__()

        # Controllo sicurezza: Dropout in LSTM funziona solo se num_layers > 1
        dropout_val = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,       # Input format: (batch, seq, feature)
            dropout=dropout_val     # Dropout interno tra i layer LSTM
        )

        # Layer finale di proiezione
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        """
        Input: (batch, seq_len, features)
        Output: (batch, features)
        """
        # Step 1: Passaggio attraverso LSTM
        # out: (batch, seq, hidden_size) -> Output per ogni passo temporale
        # _ : (h_n, c_n) -> Stati finali (non usati qui)
        out, _ = self.lstm(x)

        # Step 2: Prendiamo solo l'ultimo step temporale (Many-to-One)
        # Questo rappresenta il "riassunto" di tutta la sequenza
        last_out = out[:, -1, :] # (batch, hidden_size)

        # Step 3: Layer finale
        output = self.fc(last_out) # (batch, 9)

        return output


# ==============================================================================
# Terzo Modello - Transformer (Attention)
# ==============================================================================
class BillieTransformer(nn.Module):
    """
    Transformer Encoder
    
    - Usa Self-Attention per guardare tutta la sequenza contemporaneamente.
    - Richiede Positional Encoding per capire l'ordine.
    - Potente ma data-hungry (rischia overfitting su pochi dati).
    
    ARCHITETTURA:
    Input -> Linear Projection -> Positional Encoding -> Transformer Encoder -> Mean Pooling -> Output
    """

    def __init__(self, input_size=9, d_model=64, nhead=4, num_layers=2, max_seq_len=20, dropout=0.1):
        super(BillieTransformer, self).__init__()

        self.d_model = d_model

        # 1. Input Embedding (Proiezione lineare da 9 a 64 dim)
        self.input_projection = nn.Linear(input_size, d_model)

        # 2. Positional Encoding (Buffer pre-calcolato)
        # Creiamo un encoding fisso per la lunghezza massima prevista
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # register_buffer salva il tensore nel state_dict ma non lo aggiorna (non è un parametro)
        self.register_buffer('pe', pe.unsqueeze(0)) # (1, max_len, d_model)

        # 3. Transformer Encoder Block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Output Head
        self.fc_out = nn.Linear(d_model, input_size)

    def forward(self, x):
        """
        Input: (batch, seq_len, features)
        Output: (batch, features)
        """
        # Proiezione feature nello spazio latente (d_model)
        x = self.input_projection(x)

        # Aggiunta Positional Encoding
        # Slicing dinamico: usiamo solo la lunghezza attuale della sequenza
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]

        # Passaggio nel Transformer
        # Output: (batch, seq_len, d_model)
        x = self.transformer_encoder(x)

        # Aggregazione (Pooling)
        # Strategia: Media di tutti i vettori della sequenza (Global Average Pooling)
        # Spesso più stabile del prendere solo l'ultimo token per i Transformer
        x = x.mean(dim=1) # (batch, d_model)

        # Predizione finale
        output = self.fc_out(x) # (batch, 9)
        return output


# ==============================================================================
# TEST RAPIDO (Se eseguito come script)
# ==============================================================================
if __name__ == "__main__":
    print("="*60)
    print(" TEST ARCHITETTURE (Sanity Check)")
    print("="*60)
    
    # Parametri test
    BATCH_SIZE = 4
    SEQ_LEN = 20
    INPUT_DIM = 9
    
    # Dati dummy casuali
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)
    print(f"\nInput Shape: {dummy_input.shape} (Batch, Seq, Features)\n")
    
    # 1. Test MLP
    mlp = BillieMLP(seq_length=SEQ_LEN, input_size=INPUT_DIM, dropout=0.2)
    out_mlp = mlp(dummy_input)
    print(f" BillieMLP Output:        {out_mlp.shape} | Parametri: {sum(p.numel() for p in mlp.parameters()):,}")

    # 2. Test LSTM
    lstm = BillieLSTM(input_size=INPUT_DIM, dropout=0.2)
    out_lstm = lstm(dummy_input)
    print(f"BillieLSTM Output:       {out_lstm.shape} | Parametri: {sum(p.numel() for p in lstm.parameters()):,}")

    # 3. Test Transformer
    trans = BillieTransformer(input_size=INPUT_DIM, max_seq_len=SEQ_LEN, dropout=0.2)
    out_trans = trans(dummy_input)
    print(f" BillieTransformer Output: {out_trans.shape} | Parametri: {sum(p.numel() for p in trans.parameters()):,}")
    
    print("\nTest completato con successo. Nessun errore di dimensione.")