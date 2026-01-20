import numpy as np
from sklearn.neural_network import MLPRegressor
import joblib

class MusicOracle:
    def __init__(self):
        # Definiamo la Rete Neurale (MLP)
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 32), 
            activation='relu', 
            solver='adam', 
            warm_start=True, # Permette l'apprendimento continuo
            max_iter=1
        )
        self.loss_history = []
        self.is_trained = False

    def train_incremental(self, context_vector, target_vector):
        """Addestra l'IA sulla transizione attuale"""
        X = np.array(context_vector).reshape(1, -1)
        y = np.array(target_vector).reshape(1, -1)
        self.model.partial_fit(X, y)
        self.loss_history.append(self.model.loss_)
        self.is_trained = True

    def predict_target(self, current_context):
        """Prevede il mood della prossima canzone"""
        if not self.is_trained:
            return np.random.rand(9)
        return self.model.predict(np.array(current_context).reshape(1, -1))[0]