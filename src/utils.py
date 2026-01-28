import numpy as np
import pandas as pd

def calculate_avalanche_context(prev_context, new_song_features, n):
    """
    Calcola la media mobile solo per le feature numeriche.
    Usata per aggiornamenti incrementali rapidi.
    """
    prev_context = np.array(prev_context)
    new_song_features = np.array(new_song_features)
    
    if n <= 1:
        return new_song_features
        
    # La valanga agisce solo sui numeri (DNA acustico)
    return ((prev_context * (n - 1)) + new_song_features) / n

def calculate_weighted_context(history_df, audio_cols):
    """
    Calcola il vettore medio ponderato basato sulla colonna 'weight'.
    Fondamentale per la logica Multi-Source (Recent + Short + Long Term).
    """
    # Gestione casi limite: DF vuoto
    if history_df.empty:
        return np.array([0.5] * len(audio_cols))
        
    # Se mancano i pesi (vecchi file), fallback a media aritmetica semplice
    if 'weight' not in history_df.columns:
        return history_df[audio_cols].mean(numeric_only=True).values
    
    # Estrazione dati
    vectors = history_df[audio_cols].values
    weights = history_df['weight'].values.reshape(-1, 1)
    
    # Calcolo totale dei pesi
    total_weight = np.sum(weights)
    
    # Evitiamo divisione per zero
    if total_weight == 0:
        return np.mean(vectors, axis=0)
        
    # Calcolo media ponderata vettoriale
    # Formula: Somma(Vettore * Peso) / PesoTotale
    weighted_avg = np.sum(vectors * weights, axis=0) / total_weight
    
    return weighted_avg