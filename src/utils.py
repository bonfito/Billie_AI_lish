import numpy as np

def calculate_avalanche_context(prev_context, new_song_features, n):
    """Calcola la media mobile per il vettore contesto"""
    prev_context = np.array(prev_context)
    new_song_features = np.array(new_song_features)
    if n <= 1:
        return new_song_features
    return ((prev_context * (n - 1)) + new_song_features) / n

#aiuta il modello a ricordare le cose accadute nel passato 