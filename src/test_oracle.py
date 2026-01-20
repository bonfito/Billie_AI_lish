from oracle import MusicOracle
import numpy as np

# 1. Inizializza il tuo cervello AI
oracle = MusicOracle()

# 2. Simula una sessione: 50 canzoni con un mood coerente (es. tutte "Energiche")
# Creiamo dati che seguono un pattern logico
for i in range(50):
    context = np.random.uniform(0.5, 0.7, 9) # Mood attuale
    target = context + np.random.normal(0, 0.05, 9) # Canzone successiva (molto simile)
    
    oracle.train_incremental(context, target)
    print(f"Step {i+1} - Loss: {oracle.loss_history[-1]:.6f}")

# 3. Verifica: La loss è scesa?
if oracle.loss_history[-1] < oracle.loss_history[0]:
    print("\n✅ TEST SUPERATO: L'IA sta imparando il pattern!")
else:
    print("\n❌ TEST FALLITO: La loss non scende. Bisogna cambiare i parametri.")