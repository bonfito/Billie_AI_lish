import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.functional import cosine_similarity

import numpy as np
import random
import os
import sys
import json

# Import moduli locali
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from data_factory import create_dataloaders
from architectures import BillieMLP, BillieLSTM, BillieTransformer

# ==============================================================================
# FUNZIONI DI UTILITA E SEEDING
# ==============================================================================

def set_seed(seed=42):
    """
    Fissa i seed globali per random, numpy e torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # Rende le operazioni CUDA deterministiche (leggermente piu' lento ma preciso)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    """
    Funzione di inizializzazione per i worker del DataLoader.
    Garantisce che ogni processo di caricamento dati abbia un seed diverso ma deterministico.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================

DATA_DIR = os.path.join(current_dir, '..', 'data')
HISTORY_PATH = os.path.join(DATA_DIR, 'user_history.csv')
MODELS_DIR = os.path.join(DATA_DIR, 'trained_models')

# Crea directory se non esiste
os.makedirs(MODELS_DIR, exist_ok=True)

CONFIG = {
    'seq_length': 20,
    'batch_size': 16,
    'test_split': 0.2,
    'learning_rate': 0.001,
    'epochs': 100,       # Aumentato per permettere allo Scheduler di lavorare
    'patience': 15,      # Aumentato per tolleranza
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42
}

# ==============================================================================
# ENGINE DI TRAINING
# ==============================================================================

def train_model(model, train_dataset, test_dataset, model_name, config):
    """
    Gestisce il ciclo di vita del training:
    - Setup Dataloader deterministici
    - Training loop con gradient clipping
    - Validation e Learning Rate Scheduler
    - Early Stopping
    - Test finale
    """
    print(f"\nAVVIO TRAINING: {model_name}")
    print("-" * 50)
    
    device = config['device']
    model = model.to(device)
    
    # Setup DataLoader Deterministici
    g = torch.Generator()
    g.manual_seed(config['seed'])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    # Test loader non necessita shuffle
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Loss e Ottimizzatore
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Scheduler: Riduce il learning rate se la loss non scende per 'patience' epoche
    # FIX: rimosso verbose=True che causava errore su alcune versioni di PyTorch
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Variabili per Early Stopping
    best_test_loss = float('inf')
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'test_loss': []
    }
    
    # --- TRAINING LOOP ---
    for epoch in range(config['epochs']):
        # FASE 1: TRAINING
        model.train()
        train_losses = []
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Reset gradienti
            optimizer.zero_grad()
            
            # Forward e Loss
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            
            # Backward
            loss.backward()
            
            # Gradient Clipping (Impostato a 5.0 per stabilità LSTM/Transformer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            # Update pesi
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
        
        avg_test_loss = np.mean(test_losses) if test_losses else 0
        history['test_loss'].append(avg_test_loss)
        
        # Aggiornamento Scheduler
        # Se la test loss non migliora, riduce il learning rate
        scheduler.step(avg_test_loss)
        
        # Logging periodico
        if (epoch + 1) % 5 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoca {epoch+1:03d} | Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f} | LR: {current_lr:.6f}")
        
        # FASE 3: EARLY STOPPING E CHECKPOINT
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            patience_counter = 0
            # Salva il modello migliore
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f'{model_name}_best.pth'))
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping attivato all'epoca {epoch+1}")
                break
    
    # --- VALUTAZIONE FINALE SUL MIGLIOR MODELLO ---
    print(f"\nValutazione finale modello {model_name}...")
    
    # Ricarica pesi migliori
    model.load_state_dict(torch.load(
        os.path.join(MODELS_DIR, f'{model_name}_best.pth'),
        weights_only=True
    ))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            predictions = model(batch_x)
            
            all_preds.append(predictions.cpu())
            all_targets.append(batch_y.cpu())
    
    if all_preds:
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        mse = nn.MSELoss()(all_preds, all_targets).item()
        # Calcola similarità coseno media
        cosine = cosine_similarity(all_preds, all_targets, dim=1).mean().item()
    else:
        mse, cosine = 0.0, 0.0
    
    print(f"Risultato Finale -> MSE: {mse:.6f} | Cosine Sim: {cosine:.4f}")
    
    return {
        'model_name': model_name,
        'mse': mse,
        'cosine': cosine,
        'history': history,
        'best_epoch': epoch + 1 - patience_counter
    }

# ==============================================================================
# FUNZIONI DI OUTPUT
# ==============================================================================

def print_results(results):
    """Stampa a video i risultati di un singolo run."""
    print(f"Modello: {results['model_name']:<20} MSE: {results['mse']:.6f} Cosine: {results['cosine']:.4f}")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    # 1. Fissa seed iniziale
    set_seed(CONFIG['seed'])
    
    print("\n" + "="*60)
    print("BILLIE AI-LISH: EXPERIMENT SUITE")
    print("="*60)
    
    DATASET_SIZES = [50, 250, 500]
    all_results = []
    
    for size in DATASET_SIZES:
        print("\n" + "#"*60)
        print(f"DATASET SIZE: {size}")
        print("#"*60)
        
        # 1. Creazione Dataset (Ora restituisce Dataset, non Loader)
        # Questo permette di creare i DataLoader con seed worker qui nel main
        train_ds, test_ds, n_features = create_dataloaders(
            csv_path=HISTORY_PATH,
            seq_length=CONFIG['seq_length'],
            batch_size=CONFIG['batch_size'],
            test_split=CONFIG['test_split'],
            max_rows=size
        )
        
        # Controlli di sicurezza
        if len(train_ds) == 0 or len(test_ds) == 0:
            print(f"Attenzione: Dataset {size} troppo piccolo per la sequenza. Saltato.")
            continue
            
        train_samples = len(train_ds)
        
        # 2. Configurazione Dropout Adattivo
        # Per dataset piccoli, riduciamo il dropout per evitare underfitting
        if size <= 50:
            current_dropout = 0.0
        elif size <= 250:
            current_dropout = 0.1
        else:
            current_dropout = 0.2
            
        print(f"Info: Dropout rate impostato a {current_dropout} per size {size}")
        
        # 3. Inizializzazione Modelli
        # Passiamo il dropout calcolato ai costruttori
        models = {
            'BillieMLP': BillieMLP(seq_length=CONFIG['seq_length'], input_size=n_features, dropout=current_dropout),
            'BillieLSTM': BillieLSTM(input_size=n_features, dropout=current_dropout),
            'BillieTransformer': BillieTransformer(input_size=n_features, max_seq_len=CONFIG['seq_length'], dropout=current_dropout)
        }
        
        # 4. Training Loop
        for name, model in models.items():
            # Reset del seed prima di ogni modello per garantire equità
            set_seed(CONFIG['seed'])
            
            res = train_model(
                model=model,
                train_dataset=train_ds,
                test_dataset=test_ds,
                model_name=f"{name}_{size}",
                config=CONFIG
            )
            
            # Arricchimento dati per il report
            res['dataset_size'] = size
            res['model_base_name'] = name
            res['train_samples'] = train_samples
            
            all_results.append(res)
    
    # 5. Salvataggio Report JSON
    json_output_path = "dashboard_data.json"
    dashboard_data = []
    
    for r in all_results:
        # Formattazione storia loss
        hist_data = []
        train_h = r['history']['train_loss']
        test_h = r['history']['test_loss']
        
        for i in range(len(train_h)):
            hist_data.append({
                "Epoch": i + 1,
                "Train Loss": float(train_h[i]),
                "Test Loss": float(test_h[i])
            })
            
        dashboard_data.append({
            "Dataset": f"{r['dataset_size']} Canzoni",
            "Model": r['model_base_name'],
            "MSE": float(r['mse']),
            "Cosine": float(r['cosine']),
            "Epochs": len(train_h),
            "History": hist_data
        })
        
    try:
        with open(json_output_path, "w") as f:
            json.dump(dashboard_data, f, indent=4)
        print(f"\nDati salvati in: {os.path.abspath(json_output_path)}")
    except Exception as e:
        print(f"Errore salvataggio JSON: {e}")
        
    print("\nEsperimento completato con successo.")

if __name__ == "__main__":
    main()