import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from avalanche_logic import AvalancheAgent
from scipy.spatial import distance

# Caricamento dati
df = pd.read_csv('training_session.csv')
features = ['popularity', 'energy', 'danceability', 'valence', 'acousticness', 
            'instrumentalness', 'liveness', 'speechiness', 'loudness']

agent = AvalancheAgent()
errori = []

print("--- AVVIO SIMULAZIONE BILLIE AI-LISH ---")

for i in range(len(df) - 1):
    x_input = agent.context.copy()
    y_pred = agent.predict_next()
    y_true = df.iloc[i+1][features].values
    
    # Calcolo Errore
    err = np.mean((y_pred - y_true)**2)
    errori.append(err)
    
    # --- STEP 5: DISCOVERY (Ricerca brano più vicino) ---
    # Cerchiamo tra tutti i brani del dataset quello con minima distanza euclidea dalla previsione
    distanze = df[features].apply(lambda x: distance.euclidean(x, y_pred), axis=1)
    idx_suggerito = distanze.idxmin()
    nome_suggerito = df.iloc[idx_suggerito]['track_name']
    
    # Stampiamo un suggerimento ogni 10 brani per non intasare la console
    if i % 10 == 0:
        print(f"Brano {i} ascoltato. Prossima previsione IA: '{nome_suggerito}'")
    
    # Addestramento e Aggiornamento Contesto
    agent.train_step(x_input, y_true)
    agent.update_context(df.iloc[i][features].values)

print("--- SIMULAZIONE COMPLETATA ---")

# Grafico
plt.figure(figsize=(10, 5))
plt.plot(errori, label='Errore Predizione (MSE)')
plt.axhline(y=np.mean(errori), color='r', linestyle='--', label='Errore Medio')
plt.title("Performance Billie AI-lish: L'errore diminuisce col mood")
plt.xlabel("Sequenza Ascolti")
plt.ylabel("Errore")
plt.legend()
plt.show()