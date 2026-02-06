"""
run_experiment.py

Script per confrontare le performance di MLP, LSTM e Transformer
sulla predizione di sequenze musicali.

OUTPUT:
- Tabella Markdown con MSE e Cosine Similarity
- Grafici loss (opzionale)
- Modelli salvati su disco
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import cosine_similarity

import numpy as np
import os
import sys
import time
from collections import defaultdict

# Import moduli locali
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from data_factory import create_dataloaders
from architectures import BillieMLP, BillieLSTM, BillieTransformer


# CONFIGURAZIONE GLOBALE

# Percorsi
DATA_DIR = os.path.join(current_dir, '..', 'data')
HISTORY_PATH = os.path.join(DATA_DIR, 'user_history.csv')
MODELS_DIR = os.path.join(DATA_DIR, 'trained_models')

# Hyperparameters
CONFIG = {
    'seq_length': 20,      # Finestra temporale (20 canzoni → predici 21esima)
    'batch_size': 16,      # Ridotto per dataset piccoli (era 32)
    'test_split': 0.2,     # 20% per test
    'learning_rate': 0.001,
    'epochs': 50,          # Numero massimo di epoche
    'patience': 10,        # Early stopping (ferma se no improvement per N epoche)
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Crea cartella modelli se non esiste
os.makedirs(MODELS_DIR, exist_ok=True)


# FUNZIONE: Training di un Modello


def train_model(model, train_loader, test_loader, model_name, config):
    """
    Addestra un singolo modello e valuta le performance.
    
    Args:
        model: Modello PyTorch (BillieMLP/LSTM/Transformer)
        train_loader: DataLoader per training
        test_loader: DataLoader per test
        model_name: Nome del modello (per logging)
        config: Dizionario con hyperparameters
    
    Returns:
        dict: Contiene MSE, Cosine Similarity, Loss history
    """
    
    print(f"\n{'='*70}")
    print(f" ADDESTRAMENTO: {model_name}")
    print(f"{'='*70}")
    
    # Setup
    device = config['device']
    model = model.to(device)
    
    # Loss Function: Mean Squared Error
    # Misura quanto i valori predetti si discostano dai target
    criterion = nn.MSELoss()
    
    # Optimizer: Adam (adaptive learning rate)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Early Stopping: Variabili per monitorare miglioramenti
    best_test_loss = float('inf')
    patience_counter = 0
    
    # Storia delle loss (per grafici opzionali)
    history = {
        'train_loss': [],
        'test_loss': []
    }
    
    # TRAINING LOOP
    
    for epoch in range(config['epochs']):
        # FASE 1: TRAINING
        
        model.train()  # Abilita dropout, batch norm, ecc.
        train_losses = []
        
        for batch_x, batch_y in train_loader:
            # Sposta dati su GPU (se disponibile)
            batch_x = batch_x.to(device)  # (batch, seq, features)
            batch_y = batch_y.to(device)  # (batch, features)
            
            # STEP 1: Forward Pass
            predictions = model(batch_x)  # (batch, features)
            
            # STEP 2: Calcola Loss
            loss = criterion(predictions, batch_y)
            
            # STEP 3: Backward Pass (Backpropagation)
            optimizer.zero_grad()  # Resetta gradienti (importante!)
            loss.backward()        # Calcola gradienti
            
            # Gradient Clipping (previene gradienti esplosivi)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()       # Aggiorna pesi
            
            # Salva loss (detach per non mantenere computation graph)
            train_losses.append(loss.item())
        
        # Media loss sull'epoca
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        
        # FASE 2: VALIDATION (Test Set)
        
        model.eval()  # Disabilita dropout, ecc.
        test_losses = []
        
        with torch.no_grad():  # Disabilita gradient computation (più veloce)
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                test_losses.append(loss.item())
        
        avg_test_loss = np.mean(test_losses)
        history['test_loss'].append(avg_test_loss)
        
        # LOGGING
        
        print(f"Epoca {epoch+1:3d}/{config['epochs']} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Test Loss: {avg_test_loss:.6f}", end="")
        
        # EARLY STOPPING
        
        
        if avg_test_loss < best_test_loss:
            # Miglioramento! Salva modello
            best_test_loss = avg_test_loss
            patience_counter = 0
            
            # Salva checkpoint
            torch.save(model.state_dict(), 
                      os.path.join(MODELS_DIR, f'{model_name}_best.pth'))
            print("  [BEST]")
        else:
            # Nessun miglioramento
            patience_counter += 1
            print(f"  [Patience: {patience_counter}/{config['patience']}]")
            
            # Stop se nessun miglioramento per troppo tempo
            if patience_counter >= config['patience']:
                print(f"\n Early stopping attivato dopo {epoch+1} epoche")
                break
    
    # CARICA MIGLIOR MODELLO
    
    model.load_state_dict(
        torch.load(os.path.join(MODELS_DIR, f'{model_name}_best.pth'))
    )
    
    # 
    # VALUTAZIONE FINALE
    
    print(f"\n Valutazione finale su Test Set...")
    
    model.eval()
    all_predictions = []
    all_targets = []
    test_losses = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            
            # Accumula per metriche finali
            all_predictions.append(predictions.cpu())
            all_targets.append(batch_y.cpu())
            test_losses.append(loss.item())
    
    # Concatena tutti i batch
    all_predictions = torch.cat(all_predictions, dim=0)  # (N_test, 9)
    all_targets = torch.cat(all_targets, dim=0)          # (N_test, 9)
    
    # METRICA 1: MSE (Mean Squared Error)
    
    final_mse = np.mean(test_losses)
    
    # METRICA 2: COSINE SIMILARITY
    # Misura quanto l'"angolo" del vettore è corretto
    # 1.0 = direzione perfetta, 0.0 = perpendicolare, -1.0 = opposto
    
    cosine_sims = []
    for i in range(len(all_predictions)):
        # Calcola similarità tra vettore predetto e target
        sim = cosine_similarity(
            all_predictions[i].unsqueeze(0),  # (1, 9)
            all_targets[i].unsqueeze(0),      # (1, 9)
            dim=1
        ).item()
        cosine_sims.append(sim)
    
    avg_cosine = np.mean(cosine_sims)
    
    # RISULTATI
    
    
    results = {
        'model_name': model_name,
        'mse': final_mse,
        'cosine_similarity': avg_cosine,
        'history': history,
        'best_epoch': len(history['train_loss']) - config['patience']
    }
    
    print(f" MSE:               {final_mse:.6f}")
    print(f" Cosine Similarity: {avg_cosine:.4f}")
    
    return results


# FUNZIONE: Stampa Tabella Risultati

def print_results_table(results_list):
    """
    Stampa tabella comparativa in formato Markdown.
    
    Args:
        results_list: Lista di dizionari con risultati dei modelli
    """
    
    print("\n" + "="*70)
    print("TABELLA COMPARATIVA FINALE")
    print("="*70 + "\n")
    
    # Header
    print("| Modello            | MSE      | Cosine Similarity | Parametri  |")
    print("|--------------------|----------|-------------------|------------|")
    
    # Rows
    for res in results_list:
        model_name = res['model_name']
        mse = res['mse']
        cosine = res['cosine_similarity']
        
        # Conta parametri (dovremmo passarli, ma per ora placeholder)
        # In un'implementazione completa, li salveremmo nei risultati
        params = "N/A"
        
        print(f"| {model_name:18} | {mse:.6f} | {cosine:.4f}          | {params:10} |")
    
    print()
    
    # ANALISI VINCITORE
    
    best_mse = min(results_list, key=lambda x: x['mse'])
    best_cosine = max(results_list, key=lambda x: x['cosine_similarity'])
    
    print("VINCITORI:")
    print(f"   - Miglior MSE:               {best_mse['model_name']} ({best_mse['mse']:.6f})")
    print(f"   - Miglior Cosine Similarity: {best_cosine['model_name']} ({best_cosine['cosine_similarity']:.4f})")
    print()


# FUNZIONE MAIN
def main():
    """
    Pipeline completa dell'esperimento CON 3 DATASET DI DIMENSIONI DIVERSE.
    """
    
    print("\n" + "="*70)
    print("🎵 BILLIE AI-LISH - ESPERIMENTO COMPARATIVO MULTI-DATASET")
    print("="*70)
    
    # ═══════════════════════════════════════════════════════════════════
    # CONFIGURAZIONE: 3 dimensioni di dataset da testare
    # ═══════════════════════════════════════════════════════════════════
    
    DATASET_SIZES = [50, 250, 500]
    
    # ═══════════════════════════════════════════════════════════════════
    # LOOP PRINCIPALE: Un esperimento per ogni dimensione
    # ═══════════════════════════════════════════════════════════════════
    
    all_experiments = []  # Salverà tutti i risultati
    
    for dataset_size in DATASET_SIZES:
        
        print("\n" + "🔷"*70)
        print(f"📊 ESPERIMENTO CON ULTIME {dataset_size} CANZONI")
        print("🔷"*70)
        
        # ───────────────────────────────────────────────────────────────
        # STEP 1: CARICAMENTO DATI (con limite)
        # ───────────────────────────────────────────────────────────────
        
        print(f"\n📂 STEP 1: Caricamento ultime {dataset_size} canzoni")
        print("-" * 70)
        
        try:
            train_loader, test_loader, n_features = create_dataloaders(
                csv_path=HISTORY_PATH,
                seq_length=CONFIG['seq_length'],
                batch_size=CONFIG['batch_size'],
                test_split=CONFIG['test_split'],
                max_rows=dataset_size  # 🔥 NUOVO PARAMETRO
            )
            
            print(f"✅ Features per timestep: {n_features}")
            print(f"✅ Device: {CONFIG['device'].upper()}")
            
            # Verifica dataset non vuoto
            if len(train_loader) == 0:
                print(f"\n⚠️ SKIP: Dataset con {dataset_size} canzoni troppo piccolo per seq_length={CONFIG['seq_length']}")
                continue
            if len(test_loader) == 0:
                print(f"\n⚠️ SKIP: Test set vuoto con {dataset_size} canzoni")
                continue
                
        except FileNotFoundError:
            print(f"\n❌ ERRORE: File {HISTORY_PATH} non trovato!")
            return
        except ValueError as e:
            print(f"\n❌ ERRORE con dataset {dataset_size}: {e}")
            continue
        except Exception as e:
            print(f"\n❌ ERRORE imprevisto con dataset {dataset_size}: {e}")
            continue
        
        # ───────────────────────────────────────────────────────────────
        # STEP 2: INIZIALIZZAZIONE MODELLI
        # ───────────────────────────────────────────────────────────────
        
        print(f"\n🧠 STEP 2: Inizializzazione Modelli")
        print("-" * 70)
        
        models = {
            'BillieMLP': BillieMLP(
                seq_length=CONFIG['seq_length'],
                input_size=n_features,
                hidden_size=256
            ),
            'BillieLSTM': BillieLSTM(
                input_size=n_features,
                hidden_size=128,
                num_layers=2
            ),
            'BillieTransformer': BillieTransformer(
                input_size=n_features,
                d_model=64,
                nhead=4,
                num_layers=2
            )
        }
        
        for name, model in models.items():
            n_params = sum(p.numel() for p in model.parameters())
            print(f"✅ {name:20} {n_params:>10,} parametri")
        
        # ───────────────────────────────────────────────────────────────
        # STEP 3: TRAINING
        # ───────────────────────────────────────────────────────────────
        
        print(f"\n🚀 STEP 3: Addestramento Modelli (Dataset: {dataset_size} canzoni)")
        print("-" * 70)
        
        experiment_results = []
        
        for model_name, model in models.items():
            try:
                results = train_model(
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    model_name=f"{model_name}_{dataset_size}",  # Nome unico
                    config=CONFIG
                )
                
                # Aggiungi info sul dataset size
                results['dataset_size'] = dataset_size
                results['model_base_name'] = model_name
                
                experiment_results.append(results)
                
            except KeyboardInterrupt:
                print(f"\n⚠️ Training interrotto dall'utente")
                return
            except Exception as e:
                print(f"\n❌ ERRORE durante training di {model_name}: {e}")
                continue
        
        # ───────────────────────────────────────────────────────────────
        # STEP 4: RISULTATI PER QUESTO DATASET
        # ───────────────────────────────────────────────────────────────
        
        if experiment_results:
            print(f"\n📊 RISULTATI per Dataset {dataset_size} canzoni:")
            print_results_table(experiment_results)
            all_experiments.extend(experiment_results)
        else:
            print(f"\n⚠️ Nessun risultato per dataset {dataset_size}")
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 5: TABELLA COMPARATIVA FINALE (TUTTI I DATASET)
    # ═══════════════════════════════════════════════════════════════════
    
    if all_experiments:
        print("\n" + "="*70)
        print("📊 TABELLA COMPARATIVA FINALE - TUTTI I DATASET")
        print("="*70 + "\n")
        
        print_comparison_across_datasets(all_experiments)
    
    print("\n" + "="*70)
    print("✅ ESPERIMENTO COMPLETATO")
    print("="*70 + "\n")


# ═══════════════════════════════════════════════════════════════════════
# NUOVA FUNZIONE: Confronto tra Dataset
# ═══════════════════════════════════════════════════════════════════════

def print_comparison_across_datasets(all_results):
    """
    Stampa tabella comparativa organizzata per dataset size.
    
    Args:
        all_results: Lista di tutti i risultati (da tutti i dataset)
    """
    
    # Organizza per dataset size
    by_size = {}
    for res in all_results:
        size = res['dataset_size']
        if size not in by_size:
            by_size[size] = []
        by_size[size].append(res)
    
    # Header
    print("| Dataset Size | Modello       | MSE      | Cosine Sim | Train Samples |")
    print("|--------------|---------------|----------|------------|---------------|")
    
    # Righe organizzate per dataset size
    for size in sorted(by_size.keys()):
        results = by_size[size]
        
        for i, res in enumerate(results):
            model_name = res['model_base_name']
            mse = res['mse']
            cosine = res['cosine_similarity']
            
            # Stima train samples (approssimativa)
            train_samples = int(size * 0.8 - CONFIG['seq_length'])
            
            # Prima riga di ogni gruppo mostra dataset size
            size_col = f"{size:4d}" if i == 0 else "    "
            
            print(f"| {size_col:12} | {model_name:13} | {mse:.6f} | {cosine:.4f}     | {train_samples:13d} |")
        
        # Separatore tra dataset
        if size != max(by_size.keys()):
            print("|--------------|---------------|----------|------------|---------------|")
    
    print()
    
    # ───────────────────────────────────────────────────────────────────
    # ANALISI: Miglior modello per ogni dataset size
    # ───────────────────────────────────────────────────────────────────
    
    print("🏆 VINCITORI PER DATASET SIZE:\n")
    
    for size in sorted(by_size.keys()):
        results = by_size[size]
        
        best_mse = min(results, key=lambda x: x['mse'])
        best_cosine = max(results, key=lambda x: x['cosine_similarity'])
        
        print(f"Dataset {size:3d} canzoni:")
        print(f"  ├─ Miglior MSE:    {best_mse['model_base_name']:15} ({best_mse['mse']:.6f})")
        print(f"  └─ Miglior Cosine: {best_cosine['model_base_name']:15} ({best_cosine['cosine_similarity']:.4f})")
        print()


# ENTRY POINT


if __name__ == "__main__":
    main()