from oracle import MusicOracle
from utils import calculate_avalanche_context
import numpy as np

def test_ai_convergence():
    print("🧠 Inizializzazione Music Oracle...")
    oracle = MusicOracle()
    
    # Simuliamo un mood costante (es. musica rilassante)
    # 9 feature: energy, valence, danceability, ecc.
    current_context = np.array([0.2, 0.1, 0.2, 0.8, 0.1, 0.0, 0.1, -20.0, 0.5])
    
    print("🎧 Inizio simulazione sessione di ascolto (50 brani)...")
    
    for i in range(1, 51):
        # Generiamo una "canzone successiva" simile al mood attuale (con un po' di rumore)
        next_song = current_context + np.random.normal(0, 0.02, 9)
        
        # 1. Alleniamo l'IA sulla transizione
        oracle.train_incremental(current_context, next_song)
        
        # 2. Aggiorniamo il contesto usando la logica a Valanga (utils.py)
        current_context = calculate_avalanche_context(current_context, next_song, i)
        
        if i % 10 == 0:
            print(f"Brano {i}/50 - Loss attuale: {oracle.loss_history[-1]:.6f}")

    # Verifica finale
    first_loss = oracle.loss_history[0]
    last_loss = oracle.loss_history[-1]
    
    print("\n--- RISULTATI TEST ---")
    print(f"Loss Iniziale: {first_loss:.6f}")
    print(f"Loss Finale:   {last_loss:.6f}")
    
    if last_loss < first_loss:
        print(" SUCCESSO: L'IA sta imparando! L'errore è diminuito.")
    else:
        print(" ATTENZIONE: La loss non diminuisce. Controlla i parametri dell'MLP.")

if __name__ == "__main__":
    test_ai_convergence()