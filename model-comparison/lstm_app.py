
import pandas as pd
import os
import sys

# Import recommender
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from lstm_recommender import LSTMRecommender


class BillieAILishApp:
    """
    Applicazione interattiva per raccomandazioni musicali.
    
    Usa LSTM per predire audio features della prossima canzone,
    poi raccomanda canzoni simili dal database.
    """
    
    def __init__(self):
        """Inizializza app e carica dati."""
        
        self.data_dir = os.path.join(current_dir, '..', 'data')
        self.history_path = os.path.join(self.data_dir, 'user_history.csv')
        
        print("\n" + "="*70)
        print("🎵 BILLIE AI-LISH - LSTM Music Recommender")
        print("="*70 + "\n")
        
        # Inizializza recommender
        try:
            self.recommender = LSTMRecommender()
        except Exception as e:
            print(f"❌ Errore inizializzazione recommender: {e}")
            sys.exit(1)
        
        # Carica storico
        self.load_history()
        
        # Blacklist sessione (canzoni già consigliate)
        self.session_blacklist = []
    
    
    def load_history(self):
        """Carica storico ascolti utente."""
        
        print(f"\n📂 Caricamento storico ascolti...")
        
        if os.path.exists(self.history_path):
            try:
                self.user_history = pd.read_csv(self.history_path)
                print(f"   ✅ {len(self.user_history)} canzoni nello storico")
                
                # Mostra ultime 5
                if len(self.user_history) >= 5:
                    print(f"\n   Ultime 5 canzoni ascoltate:")
                    for idx, row in self.user_history.tail(5).iterrows():
                        name = row.get('name', 'Unknown')
                        artist = row.get('artist', 'Unknown')
                        print(f"   - {name} - {artist}")
                
            except Exception as e:
                print(f"   ⚠️  Errore caricamento: {e}")
                self.user_history = pd.DataFrame()
        else:
            print(f"   ⚠️  Nessuno storico trovato")
            print(f"   Usa predizione neutra (0.5 per tutte le features)")
            self.user_history = pd.DataFrame()
    
    
    def get_recommendations(self, k=20):
        """
        Ottiene K raccomandazioni.
        
        Args:
            k (int): Numero raccomandazioni
        
        Returns:
            pd.DataFrame: Raccomandazioni
        """
        
        recommendations, predicted = self.recommender.recommend(
            user_history_df=self.user_history,
            k=k,
            exclude_listened=True,
            session_blacklist=self.session_blacklist
        )
        
        # Aggiungi alla blacklist sessione
        if not recommendations.empty and 'id' in recommendations.columns:
            new_ids = recommendations['id'].tolist()
            self.session_blacklist.extend(new_ids)
        
        return recommendations, predicted
    
    
    def display_recommendations(self, recommendations):
        """
        Mostra raccomandazioni all'utente.
        
        Args:
            recommendations (pd.DataFrame): Raccomandazioni
        """
        
        if recommendations.empty:
            print("\n❌ Nessuna raccomandazione disponibile")
            return
        
        print("\n" + "="*70)
        print("🎵 RACCOMANDAZIONI PER TE")
        print("="*70 + "\n")
        
        for idx, row in recommendations.iterrows():
            rank = row.get('rank', idx + 1)
            name = row.get('name', 'Unknown')
            artist = row.get('artist', 'Unknown')
            score = row.get('match_percentage', 0)
            
            # Features principali (se disponibili)
            energy = row.get('energy', None)
            valence = row.get('valence', None)
            
            print(f"{rank:2d}. {name[:45]:<45}")
            print(f"    Artista: {artist[:50]}")
            print(f"    Match: {score:.1f}%", end="")
            
            if energy is not None and valence is not None:
                mood = self._get_mood(energy, valence)
                print(f" | Mood: {mood}", end="")
            
            print("\n")
    
    
    def _get_mood(self, energy, valence):
        """
        Determina mood da energy e valence.
        
        Quadranti:
        - Alta energy, alta valence: Felice/Energico
        - Alta energy, bassa valence: Arrabbiato/Intenso
        - Bassa energy, alta valence: Calmo/Rilassato
        - Bassa energy, bassa valence: Triste/Malinconico
        
        Args:
            energy (float): [0-1]
            valence (float): [0-1]
        
        Returns:
            str: Descrizione mood
        """
        
        if energy > 0.6 and valence > 0.6:
            return "Energico 🔥"
        elif energy > 0.6 and valence < 0.4:
            return "Intenso ⚡"
        elif energy < 0.4 and valence > 0.6:
            return "Rilassante 🌊"
        elif energy < 0.4 and valence < 0.4:
            return "Malinconico 🌧️"
        else:
            return "Neutro 😌"
    
    
    def interactive_mode(self):
        """Modalità interattiva con menu."""
        
        while True:
            print("\n" + "="*70)
            print("MENU")
            print("="*70)
            print("1. Ottieni raccomandazioni (20 canzoni)")
            print("2. Ottieni raccomandazioni (10 canzoni)")
            print("3. Visualizza storico")
            print("4. Analizza predizione LSTM")
            print("5. Reset blacklist sessione")
            print("0. Esci")
            print("="*70)
            
            choice = input("\nScelta: ").strip()
            
            if choice == '1':
                recs, _ = self.get_recommendations(k=20)
                self.display_recommendations(recs)
            
            elif choice == '2':
                recs, _ = self.get_recommendations(k=10)
                self.display_recommendations(recs)
            
            elif choice == '3':
                self.show_history()
            
            elif choice == '4':
                self.analyze_prediction()
            
            elif choice == '5':
                self.session_blacklist = []
                print("\n✅ Blacklist resettata")
            
            elif choice == '0':
                print("\n👋 Grazie per aver usato Billie AI-lish!\n")
                break
            
            else:
                print("\n❌ Scelta non valida")
    
    
    def show_history(self):
        """Mostra storico ascolti."""
        
        if self.user_history.empty:
            print("\n⚠️  Nessuno storico disponibile")
            return
        
        print("\n" + "="*70)
        print("📜 STORICO ASCOLTI")
        print("="*70 + "\n")
        
        print(f"Totale canzoni: {len(self.user_history)}")
        print(f"\nUltime 20 canzoni:")
        print("─"*70)
        
        for idx, row in self.user_history.tail(20).iterrows():
            name = row.get('name', 'Unknown')
            artist = row.get('artist', 'Unknown')
            print(f"{len(self.user_history) - idx}. {name} - {artist}")
    
    
    def analyze_prediction(self):
        """Analizza predizione LSTM."""
        
        print("\n" + "="*70)
        print("📊 ANALISI PREDIZIONE LSTM")
        print("="*70 + "\n")
        
        analysis = self.recommender.analyze_prediction(self.user_history)
        
        if analysis['history_mean'] is None:
            print("⚠️  Nessuno storico disponibile per l'analisi")
            return
        
        print("Confronto: Media Storico vs Predizione LSTM\n")
        print(f"{'Feature':<20} {'Storico':<12} {'Predizione':<12} {'Trend':<10}")
        print("─"*70)
        
        for i, feat in enumerate(analysis['feature_names']):
            hist = analysis['history_mean'][i]
            pred = analysis['predicted'][i]
            diff = analysis['difference'][i]
            
            if abs(diff) < 0.05:
                trend = "→ Stabile"
            elif diff > 0:
                trend = f"↑ +{diff:.3f}"
            else:
                trend = f"↓ {diff:.3f}"
            
            print(f"{feat:<20} {hist:>6.3f}       {pred:>6.3f}       {trend}")
        
        print("\n" + "="*70)
        
        # Interpretazione
        print("\n💡 INTERPRETAZIONE:")
        
        if 'energy' in analysis['feature_names']:
            idx = analysis['feature_names'].index('energy')
            energy_diff = analysis['difference'][idx]
            
            if energy_diff > 0.1:
                print("   🔥 LSTM prevede un AUMENTO di energia")
                print("   → Raccomandazioni più energiche del solito")
            elif energy_diff < -0.1:
                print("   🌊 LSTM prevede una DIMINUZIONE di energia")
                print("   → Raccomandazioni più rilassate del solito")
            else:
                print("   😌 LSTM prevede energia STABILE")
                print("   → Raccomandazioni simili allo storico")
        
        if 'valence' in analysis['feature_names']:
            idx = analysis['feature_names'].index('valence')
            valence_diff = analysis['difference'][idx]
            
            if valence_diff > 0.1:
                print("   😊 LSTM prevede un AUMENTO di positività")
                print("   → Raccomandazioni più allegre")
            elif valence_diff < -0.1:
                print("   😔 LSTM prevede una DIMINUZIONE di positività")
                print("   → Raccomandazioni più malinconiche")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    """Entry point applicazione."""
    
    try:
        app = BillieAILishApp()
        app.interactive_mode()
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interruzione utente")
        print("👋 Arrivederci!\n")
    
    except Exception as e:
        print(f"\n❌ Errore critico: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()