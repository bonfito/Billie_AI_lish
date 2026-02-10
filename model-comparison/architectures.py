import torch
import torch.nn as nn

import math

#Primo Modello - MLP
class BillieMLP(nn.Module):
    """
    - Appiattisce la sequenza in un vettore unico
    - Non capisce l'ordine temporale

    ARCHITETTURA
    INPUT(20,9) --> Flatten (180) --> Dense (256) --> ReLU --> Dense (128) --> ReLU --> Output(9)
    """

    def __init__(self, seq_length=20, input_size=9, hidden_size=256):
        super(BillieMLP, self).__init__()

        #dimensione dopo il flatten: seq_length * input_size
        flattened_size = seq_length * input_size #20 * 9 = 180

        #layer densi fully connected

        #creazione di una matrice di pesi per collegare i neuroni dei diversi layer
        #in questa fase le matrici vengono riempite con numeri molto piccoli, casuali
        #semplicemente perché ancora la RN è appena nata, non ha mai 'visto' musica
        self.fc1 = nn.Linear(flattened_size, hidden_size) #180 -> 256
        self.fc2 = nn.Linear(hidden_size, hidden_size//2) #256 -> 128
        self.fc3 = nn.Linear(hidden_size // 2, input_size) #128 -> 9

        #funzione di attivazione ReLU standard
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass

        Argomenti:
            x: Tensor (batch_size, seq_length, input_size)
            Esempio (32,20,9)

        Ritorna:

            y: tensor (bacth_size, input_size)
            Esempio (32,9)
        """

        #step 1 - Flatten, appiatisce intera sequenza
        #passiamo da (32,20,9) a (32,180)

        batch_size = x.size(0)
        x = x.view(batch_size, -1) # -1 , calcola automaticamente (180)

        #step 2 - passaggio attraverso layer densi
        x = self.relu(self.fc1(x))  # (32, 180) -> (32, 256)
        x = self.relu(self.fc2(x))  # (32, 256) -> (32, 128)
        x = self.fc3(x)              # (32, 128) -> (32, 9)

        return x
    
#Secondo Modello - LSTM 
class BillieLSTM(nn.Module):
    """
    Long Short-Term Memory

    - legge le canzoni sequenzialmente
    - Mantiene una memoria interna che si aggiorna ad ogni timestamp
    - comprende quelli che sono i trend temporali

    ARCHITETTURA
    Input (20, 9) → LSTM(hidden=128, layers=2) → Ultimo Hidden State → Dense(9)

    """

    def __init__(self, input_size = 9, hidden_size = 128, num_layers = 2):
        super(BillieLSTM, self).__init__()

        #LSTM Layer
        #input size è il numero di feature per ogni timestamp (9)
        #hidden size è la dimensione della memoria interna (128)
        #num_layers: Numero di LSTM impilati (2 = più profondo)
        # batch_first=True: Input sarà (batch, seq, features) invece di (seq, batch, features)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout= 0.2 #per evitare overfitting
        )

        #Layer finale per output
        self.fc = nn.Linear(hidden_size, input_size) #128 -> 9

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Tensor (batch_size, seq_length, input_size)
               Esempio: (32, 20, 9)
        
        Returns:
            output: Tensor (batch_size, input_size)
                    Esempio: (32, 9)
        """

        #step 1 - passaggio attraverso LSTM
        # lstm_out: (batch, seq, hidden_size) = (32, 20, 128)
        #   Contiene hidden state per OGNI timestep
        # h_n: (num_layers, batch, hidden_size) = (2, 32, 128)
        #   Hidden state FINALE (dopo aver letto tutte le 20 canzoni)
        # c_n: Cell state (non ci serve) 

        lstm_out, (h_n, c_n) = self.lstm(x)

        #step 2 - estrazione ultimo hidden state
        # h_n[-1] = ultimo layer ->il + alto nello stack
        #Shape (batch, hidden_size) = (32,128)

        last_hidden = h_n[-1]

        #step 3 --> passaggio attraverso layer finale
        output = self.fc(last_hidden) # 32,128 -> 32,9

        return output
    
#Terzo Modello - Transformer (Attention)
class BillieTransformer(nn.Module):
    """
    - Ogni canzone guarda le altre contemporaneamente
    - Meccanismo di attenzione: "Quale canzone è più rilevante?"
    - Può dare peso a canzoni lontane nella sessione (ad esempio la prima)
    
    ARCHITETTURA:
    Input (20, 9) → Positional Encoding → Transformer Encoder → Avg Pool → Dense(9)

    """

    def __init__(self, input_size=9, d_model=64, nhead=4, num_layers=2):
        super(BillieTransformer, self).__init__()

        # PARAMETRI:
        # input_size: Feature per timestep (9)
        # d_model: Dimensione embedding interna (deve essere divisibile per nhead)
        # nhead: Numero di "teste" di attenzione (4 = guarda da 4 prospettive diverse)
        # num_layers: Numero di Transformer Encoder impilati

        self.d_model = d_model

        #STEP 1 - Proiezione Input
        # Le 9 Feature vengono implementate in uno spazio d_model dimensionale
        self.input_projection = nn.Linear(input_size, d_model) #9 -> 64

        #STEP 2 - Positional Encoding
        # Transformer non conosce ordine -> dobbiamo dire quale posizione ha ogni canzone
        # si usa un buffer con encoding sinusoidale (non trainable)

        self.register_buffer('positional_encoding', self._create_positional_encoding(max_len=100))

        # STEP 3 - Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4, #64 * 4 = 256 (standard)
            dropout=0.1,
            batch_first=True #input (batch, seq, features)
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        #STEP 4: output layer
        self.fc = nn.Linear(d_model, input_size) #64 -> 9

    def _create_positional_encoding(self, max_len=100):
        """
        Crea Positional Encoding sinusoidale.
        
        FORMULA (dal paper "Attention is All You Need"):
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        
        Args:
            max_len: Lunghezza massima sequenza supportata
        
        Returns:
            pe: Tensor (1, max_len, d_model)
        """
        pe = torch.zeros(max_len, self.d_model)
        
        # Posizioni: [0, 1, 2, ..., 99]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (100, 1)
        
        # Divisori per frequenze sinusoidali
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float() * 
            (-math.log(10000.0) / self.d_model)
        )
        
        # Applica sin alle posizioni pari, cos alle posizioni dispari
        pe[:, 0::2] = torch.sin(position * div_term)  # Colonne 0, 2, 4, ...
        pe[:, 1::2] = torch.cos(position * div_term)  # Colonne 1, 3, 5, ...
        
        return pe.unsqueeze(0)  # (1, max_len, d_model)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Tensor (batch_size, seq_length, input_size)
               Esempio: (32, 20, 9)
        
        Returns:
            output: Tensor (batch_size, input_size)
                    Esempio: (32, 9)
        """
        batch_size, seq_len, _ = x.size()
        
        # STEP 1: Proiezione Input
        # (32, 20, 9) → (32, 20, 64)
        x = self.input_projection(x)
        
        # STEP 2: Aggiungi Positional Encoding
        # Prendi solo i primi seq_len encoding (20 nel nostro caso)
        pos_enc = self.positional_encoding[:, :seq_len, :].to(x.device)
        x = x + pos_enc  # Broadcast automatico su batch
        
        # STEP 3: Passa attraverso Transformer
        # (32, 20, 64) → (32, 20, 64)
        # Internamente applica Self-Attention + Feedforward
        x = self.transformer(x)
        
        # STEP 4: Aggregazione
        # Abbiamo output per ogni timestep (20)
        # Dobbiamo ridurre a un singolo vettore
        
        # OPZIONE A: Prendi solo l'ultimo timestep (come LSTM)
        # x = x[:, -1, :]  # (32, 64)
        
        # OPZIONE B: Media di tutti i timestep (migliore per Transformer)
        x = x.mean(dim=1)  # (32, 20, 64) → (32, 64)
        
        # STEP 5: Layer finale
        output = self.fc(x)  # (32, 64) → (32, 9)
        
        return output
    
#TEST ARCHITETTURE

if __name__ == "__main__":
    print("="*60)
    print("TEST ARCHITETTURE")
    print("="*60)
    
    # Crea input dummy (batch=4, seq=20, features=9)
    dummy_input = torch.randn(4, 20, 9)
    
    print(f"\nInput Shape: {dummy_input.shape}")
    print(f"Rappresenta: 4 esempi, 20 canzoni ciascuno, 9 feature per canzone\n")
    
    # Test 1: MLP
    print("─" * 60)
    print("TEST 1: BillieMLP")
    print("─" * 60)
    
    mlp = BillieMLP(seq_length=20, input_size=9, hidden_size=256)
    mlp_output = mlp(dummy_input)
    
    print(f"Output Shape: {mlp_output.shape}")
    print(f"Parametri totali: {sum(p.numel() for p in mlp.parameters()):,}")
    print(f"Esempio output: {mlp_output[0, :3]}")  # Prime 3 feature del primo esempio
    
    # Test 2: LSTM
    print("\n" + "─" * 60)
    print("TEST 2: BillieLSTM")
    print("─" * 60)
    
    lstm = BillieLSTM(input_size=9, hidden_size=128, num_layers=2)
    lstm_output = lstm(dummy_input)
    
    print(f"Output Shape: {lstm_output.shape}")
    print(f"Parametri totali: {sum(p.numel() for p in lstm.parameters()):,}")
    print(f"Esempio output: {lstm_output[0, :3]}")
    
     
    # Test 3: Transformer
    print("\n" + "─" * 60)
    print("TEST 3: BillieTransformer")
    print("─" * 60)
    
    transformer = BillieTransformer(input_size=9, d_model=64, nhead=4, num_layers=2)
    transformer_output = transformer(dummy_input)
    
    print(f"Output Shape: {transformer_output.shape}")
    print(f"Parametri totali: {sum(p.numel() for p in transformer.parameters()):,}")
    print(f"Esempio output: {transformer_output[0, :3]}")
    
    # Confronto Complessità
    print("\n" + "="*60)
    print("CONFRONTO COMPLESSITÀ MODELLI")
    print("="*60)
    
    models = {
        "BillieMLP": mlp,
        "BillieLSTM": lstm,
        "BillieTransformer": transformer
    }
    
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:20} {params:>10,} parametri")
    
    print("\nTutti i modelli funzionano correttamente!")
    

