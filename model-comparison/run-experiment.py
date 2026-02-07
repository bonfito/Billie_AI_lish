"""
run_experiment.py - VERSIONE CON RIPRODUCIBILITÀ

Script per confrontare le performance di MLP, LSTM e Transformer
sulla predizione di sequenze musicali.

NOVITÀ: Seed fissi per risultati riproducibili!

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
import random
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


# ═══════════════════════════════════════════════════════════════════════════
# FUNZIONE: Fissa Seed per Riproducibilità
# ═══════════════════════════════════════════════════════════════════════════

def set_seed(seed=42):
    """
    Fissa tutti i seed random per garantire risultati riproducibili.
    
    IMPORTANTE: Questa funzione VA CHIAMATA PRIMA di creare modelli o dataset!
    
    Args:
        seed (int): Valore del seed (default: 42)
    
    Cosa fissa:
    - random (Python standard library)
    - numpy.random
    - torch.manual_seed (CPU)
    - torch.cuda.manual_seed (GPU)
    - torch.backends.cudnn (operazioni CUDA)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Per multi-GPU
        
    # Rende le operazioni CUDA deterministiche
    # NOTA: Può rallentare leggermente il training (~5-10%)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"🎲 Seed fissato a: {seed}")
    print("   ✅ Risultati saranno identici ad ogni esecuzione")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURAZIONE GLOBALE
# ═══════════════════════════════════════════════════════════════════════════

# Percorsi
DATA_DIR = os.path.join(current_dir, '..', 'data')
HISTORY_PATH = os.path.join(DATA_DIR, 'user_history.csv')
MODELS_DIR = os.path.join(DATA_DIR, 'trained_models')

# Hyperparameters
CONFIG = {
    'seq_length': 20,      # Finestra temporale (20 canzoni → predici 21esima)
    'batch_size': 16,      # Ridotto per dataset piccoli
    'test_split': 0.2,     # 20% per test
    'learning_rate': 0.001,
    'epochs': 50,          # Numero massimo di epoche
    'patience': 10,        # Early stopping
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42             # 🎲 SEED FISSO (cambialo se vuoi risultati diversi)
}

# Crea cartella per modelli salvati
os.makedirs(MODELS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# FUNZIONE: Training di un Modello
# ═══════════════════════════════════════════════════════════════════════════

def train_model(model, train_loader, test_loader, model_name, config):
    """
    Addestra un singolo modello e valuta le performance.
    """
    
    print(f"\n{'='*70}")
    print(f"🚀 ADDESTRAMENTO: {model_name}")
    print(f"{'='*70}")
    
    # Setup
    device = config['device']
    model = model.to(device)
    
    # Loss Function
    criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Early Stopping
    best_test_loss = float('inf')
    patience_counter = 0
    
    # Storia delle loss
    history = {
        'train_loss': [],
        'test_loss': []
    }
    
    # ───────────────────────────────────────────────────────────────────────
    # TRAINING LOOP
    # ───────────────────────────────────────────────────────────────────────
    
    for epoch in range(config['epochs']):
        # FASE 1: TRAINING
        model.train()
        train_losses = []
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Forward Pass
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            
            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping (evita esplosione gradienti)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        
        # FASE 2: VALIDATION
        model.eval()
        test_losses = []
        
        with torch.no_grad():
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
            best_test_loss = avg_test_loss
            patience_counter = 0
            
            # Salva checkpoint
            torch.save(model.state_dict(), 
                      os.path.join(MODELS_DIR, f'{model_name}_best.pth'))
            print(" ✅ [BEST]")
        else:
            patience_counter += 1
            print(f" ⏳ [Patience: {patience_counter}/{config['patience']}]")
            
            if patience_counter >= config['patience']:
                print(f"\n⚠️ Early stopping attivato dopo {epoch+1} epoche")
                break
    
    # ───────────────────────────────────────────────────────────────────────
    # VALUTAZIONE FINALE
    # ───────────────────────────────────────────────────────────────────────
    
    # Ricarica il miglior modello
    model.load_state_dict(torch.load(
        os.path.join(MODELS_DIR, f'{model_name}_best.pth'),
        weights_only=True
    ))
    
    print(f"\n📊 Valutazione finale su Test Set...")
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            predictions = model(batch_x)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(batch_y.cpu())
    
    # Concatena tutti i batch
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calcola metriche
    mse = nn.MSELoss()(all_predictions, all_targets).item()
    
    # Cosine Similarity (media su tutti gli esempi)
    cosine_sims = cosine_similarity(all_predictions, all_targets, dim=1)
    avg_cosine = cosine_sims.mean().item()
    
    print(f" ✅ MSE:               {mse:.6f}")
    print(f" ✅ Cosine Similarity: {avg_cosine:.4f}")
    
    return {
        'model_name': model_name,
        'mse': mse,
        'cosine': avg_cosine,
        'history': history,
        'best_epoch': epoch + 1 - patience_counter
    }


# ═══════════════════════════════════════════════════════════════════════════
# FUNZIONE: Stampa Tabella Comparativa
# ═══════════════════════════════════════════════════════════════════════════

def print_comparison_table(results_list, dataset_size):
    """
    Stampa tabella Markdown con risultati.
    """
    
    print(f"\n📊 RISULTATI per Dataset {dataset_size} canzoni:\n")
    
    print("="*70)
    print("TABELLA COMPARATIVA FINALE")
    print("="*70)
    print()
    
    # Header
    print(f"| {'Modello':<18} | {'MSE':<8} | {'Cosine Similarity':<17} | {'Parametri':<10} |")
    print(f"|{'-'*20}|{'-'*10}|{'-'*19}|{'-'*12}|")
    
    # Righe dati
    for res in results_list:
        print(f"| {res['model_name']:<18} | {res['mse']:<8.6f} | {res['cosine']:<17.4f} | {'N/A':<10} |")
    
    print()
    
    # Trova vincitori
    best_mse_model = min(results_list, key=lambda x: x['mse'])
    best_cosine_model = max(results_list, key=lambda x: x['cosine'])
    
    print("VINCITORI:")
    print(f"   - Miglior MSE:               {best_mse_model['model_name']} ({best_mse_model['mse']:.6f})")
    print(f"   - Miglior Cosine Similarity: {best_cosine_model['model_name']} ({best_cosine_model['cosine']:.4f})")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# FUNZIONE: Stampa Tabella Multi-Dataset
# ═══════════════════════════════════════════════════════════════════════════

def print_multi_dataset_table(all_experiments):
    """
    Stampa tabella comparativa su tutti i dataset testati.
    """
    
    print("\n" + "="*70)
    print("📊 TABELLA COMPARATIVA FINALE - TUTTI I DATASET")
    print("="*70)
    print()
    
    # Header
    print(f"| {'Dataset Size':<12} | {'Modello':<15} | {'MSE':<8} | {'Cosine Sim':<11} | {'Train Samples':<14} |")
    print(f"|{'-'*14}|{'-'*17}|{'-'*10}|{'-'*13}|{'-'*16}|")
    
    # Organizza per dataset size
    by_size = {}
    for exp in all_experiments:
        size = exp['dataset_size']
        if size not in by_size:
            by_size[size] = []
        by_size[size].append(exp)
    
    # Stampa
    for size in sorted(by_size.keys()):
        experiments = by_size[size]
        
        for i, exp in enumerate(experiments):
            size_str = str(size) if i == 0 else ""
            model_str = exp['model_base_name']
            
            print(f"| {size_str:<12} | {model_str:<15} | {exp['mse']:<8.6f} | {exp['cosine']:<11.4f} | {exp['train_samples']:<14} |")
        
        # Separatore tra dataset
        print(f"|{'-'*14}|{'-'*17}|{'-'*10}|{'-'*13}|{'-'*16}|")
    
    # Vincitori per ogni dataset
    print("\n🏆 VINCITORI PER DATASET SIZE:\n")
    
    for size in sorted(by_size.keys()):
        experiments = by_size[size]
        best_mse = min(experiments, key=lambda x: x['mse'])
        best_cosine = max(experiments, key=lambda x: x['cosine'])
        
        print(f"Dataset {size:3d} canzoni:")
        print(f"  ├─ Miglior MSE:    {best_mse['model_base_name']:<20} ({best_mse['mse']:.6f})")
        print(f"  └─ Miglior Cosine: {best_cosine['model_base_name']:<20} ({best_cosine['cosine']:.4f})")
        print()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN: Esecuzione Esperimento
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """
    Funzione principale: esegue esperimento su 3 dataset sizes.
    """
    
    # ═══════════════════════════════════════════════════════════════════════
    # 🎲 PASSO 1: FISSA IL SEED (CRUCIALE!)
    # ═══════════════════════════════════════════════════════════════════════
    
    set_seed(CONFIG['seed'])
    
    # ═══════════════════════════════════════════════════════════════════════
    # INTESTAZIONE
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n" + "="*70)
    print("🎵 BILLIE AI-LISH - ESPERIMENTO COMPARATIVO MULTI-DATASET")
    print("="*70)
    print()
    
    # Dataset sizes da testare
    DATASET_SIZES = [50, 250, 500]
    
    all_experiments = []
    
    # ═══════════════════════════════════════════════════════════════════════
    # LOOP SU DATASET SIZES
    # ═══════════════════════════════════════════════════════════════════════
    
    for dataset_size in DATASET_SIZES:
        
        print("\n" + "🔷"*70)
        print(f"📊 ESPERIMENTO CON ULTIME {dataset_size} CANZONI")
        print("🔷"*70 + "\n")
        
        # ───────────────────────────────────────────────────────────────────
        # STEP 1: Caricamento Dati
        # ───────────────────────────────────────────────────────────────────
        
        print(f"📂 STEP 1: Caricamento ultime {dataset_size} canzoni")
        print("-" * 70)
        
        train_loader, test_loader, n_features = create_dataloaders(
            csv_path=HISTORY_PATH,
            seq_length=CONFIG['seq_length'],
            batch_size=CONFIG['batch_size'],
            test_split=CONFIG['test_split'],
            max_rows=dataset_size
        )
        
        print(f"✅ Features per timestep: {n_features}")
        print(f"✅ Device: {CONFIG['device']}")
        print()
        
        # Train samples (per statistiche)
        train_samples = len(train_loader.dataset)
        
        # ───────────────────────────────────────────────────────────────────
        # STEP 2: Inizializzazione Modelli
        # ───────────────────────────────────────────────────────────────────
        
        print("🧠 STEP 2: Inizializzazione Modelli")
        print("-" * 70)
        
        # 🎲 IMPORTANTE: Fissa seed PRIMA di creare ogni modello
        # Così i pesi iniziali sono identici ad ogni esecuzione
        set_seed(CONFIG['seed'])
        
        models = {
            'BillieMLP': BillieMLP(seq_length=CONFIG['seq_length'], input_size=n_features),
            'BillieLSTM': BillieLSTM(input_size=n_features),
            'BillieTransformer': BillieTransformer(input_size=n_features)
        }
        
        for name, model in models.items():
            params = sum(p.numel() for p in model.parameters())
            print(f"✅ {name:<20} {params:>10,} parametri")
        
        print()
        
        # ───────────────────────────────────────────────────────────────────
        # STEP 3: Training
        # ───────────────────────────────────────────────────────────────────
        
        print(f"🚀 STEP 3: Addestramento Modelli (Dataset: {dataset_size} canzoni)")
        print("-" * 70)
        
        results_for_this_dataset = []
        
        for model_name, model in models.items():
            # 🎲 Fissa seed prima di ogni training per consistenza
            set_seed(CONFIG['seed'])
            
            results = train_model(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                model_name=f"{model_name}_{dataset_size}",
                config=CONFIG
            )
            
            # Aggiungi info dataset
            results['dataset_size'] = dataset_size
            results['model_base_name'] = model_name
            results['train_samples'] = train_samples
            
            results_for_this_dataset.append(results)
            all_experiments.append(results)
        
        # ───────────────────────────────────────────────────────────────────
        # Tabella risultati per questo dataset
        # ───────────────────────────────────────────────────────────────────
        
        print_comparison_table(results_for_this_dataset, dataset_size)
    
    # ═══════════════════════════════════════════════════════════════════════
    # TABELLA FINALE MULTI-DATASET
    # ═══════════════════════════════════════════════════════════════════════
    
    print_multi_dataset_table(all_experiments)
    
    print("\n" + "="*70)
    print("✅ ESPERIMENTO COMPLETATO")
    print("="*70)
    print()


if __name__ == "__main__":
    main()