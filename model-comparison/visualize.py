# -*- coding: utf-8 -*-
"""
visualize.py - VISUALIZZAZIONE RISULTATI ESPERIMENTO

Genera grafici e tabelle professionali per la presentazione dei risultati.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path

# Configurazione stile professionale
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


# ═══════════════════════════════════════════════════════════════════════════
# DATI SPERIMENTALI (dai risultati)
# ═══════════════════════════════════════════════════════════════════════════

RESULTS = {
    50: {
        'BillieMLP': {'mse': 0.024614, 'cosine': 0.9610, 'params': 80393, 'epochs': 39},
        'BillieLSTM': {'mse': 0.024448, 'cosine': 0.9612, 'params': 204425, 'epochs': 43},
        'BillieTransformer': {'mse': 0.026381, 'cosine': 0.9594, 'params': 101193, 'epochs': 21}
    },
    250: {
        'BillieMLP': {'mse': 0.026512, 'cosine': 0.9452, 'params': 80393, 'epochs': 16},
        'BillieLSTM': {'mse': 0.026717, 'cosine': 0.9464, 'params': 204425, 'epochs': 19},
        'BillieTransformer': {'mse': 0.027695, 'cosine': 0.9468, 'params': 101193, 'epochs': 17}
    },
    500: {
        'BillieMLP': {'mse': 0.026804, 'cosine': 0.9521, 'params': 80393, 'epochs': 13},
        'BillieLSTM': {'mse': 0.026522, 'cosine': 0.9519, 'params': 204425, 'epochs': 17},
        'BillieTransformer': {'mse': 0.026695, 'cosine': 0.9519, 'params': 101193, 'epochs': 23}
    }
}

# Colori per i modelli
COLORS = {
    'BillieMLP': '#FF6B6B',          # Rosso
    'BillieLSTM': '#4ECDC4',         # Turchese
    'BillieTransformer': '#FFD93D'   # Giallo
}


# ═══════════════════════════════════════════════════════════════════════════
# GRAFICO 1: MSE per Dataset Size
# ═══════════════════════════════════════════════════════════════════════════

def plot_mse_comparison():
    """
    Grafico a barre raggruppate: MSE per ogni modello su ogni dataset.
    """
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    dataset_sizes = [50, 250, 500]
    models = ['BillieMLP', 'BillieLSTM', 'BillieTransformer']
    
    x = np.arange(len(dataset_sizes))
    width = 0.25
    
    for i, model in enumerate(models):
        mse_values = [RESULTS[size][model]['mse'] for size in dataset_sizes]
        offset = width * (i - 1)
        bars = ax.bar(x + offset, mse_values, width, 
                      label=model, color=COLORS[model], alpha=0.8, edgecolor='black')
        
        # Aggiungi valori sopra le barre
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.5f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Dataset Size (numero di canzoni)', fontweight='bold', fontsize=12)
    ax.set_ylabel('MSE (Mean Squared Error)', fontweight='bold', fontsize=12)
    ax.set_title('Confronto MSE tra Modelli per Dataset Size\n(Valori più bassi = Migliore)', 
                 fontweight='bold', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{size} canzoni' for size in dataset_sizes])
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    
    # Evidenzia il miglior modello per ogni dataset
    for idx, size in enumerate(dataset_sizes):
        best_model = min(models, key=lambda m: RESULTS[size][m]['mse'])
        best_mse = RESULTS[size][best_model]['mse']
        model_idx = models.index(best_model)
        ax.plot(idx + width * (model_idx - 1), best_mse, 'g*', markersize=15, 
                markeredgecolor='black', markeredgewidth=1.5)
    
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# GRAFICO 2: Cosine Similarity per Dataset Size
# ═══════════════════════════════════════════════════════════════════════════

def plot_cosine_comparison():
    """
    Grafico a barre raggruppate: Cosine Similarity per ogni modello.
    """
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    dataset_sizes = [50, 250, 500]
    models = ['BillieMLP', 'BillieLSTM', 'BillieTransformer']
    
    x = np.arange(len(dataset_sizes))
    width = 0.25
    
    for i, model in enumerate(models):
        cosine_values = [RESULTS[size][model]['cosine'] for size in dataset_sizes]
        offset = width * (i - 1)
        bars = ax.bar(x + offset, cosine_values, width, 
                      label=model, color=COLORS[model], alpha=0.8, edgecolor='black')
        
        # Aggiungi valori sopra le barre
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Dataset Size (numero di canzoni)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontweight='bold', fontsize=12)
    ax.set_title('Confronto Cosine Similarity tra Modelli per Dataset Size\n(Valori più alti = Migliore)', 
                 fontweight='bold', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{size} canzoni' for size in dataset_sizes])
    ax.set_ylim(0.93, 0.97)  # Focus sulla zona di interesse
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    
    # Evidenzia il miglior modello per ogni dataset
    for idx, size in enumerate(dataset_sizes):
        best_model = max(models, key=lambda m: RESULTS[size][m]['cosine'])
        best_cosine = RESULTS[size][best_model]['cosine']
        model_idx = models.index(best_model)
        ax.plot(idx + width * (model_idx - 1), best_cosine, 'g*', markersize=15,
                markeredgecolor='black', markeredgewidth=1.5)
    
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# GRAFICO 3: Trend MSE all'Aumentare del Dataset
# ═══════════════════════════════════════════════════════════════════════════

def plot_mse_trends():
    """
    Grafico a linee: Come cambia MSE all'aumentare del dataset.
    """
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    dataset_sizes = [50, 250, 500]
    models = ['BillieMLP', 'BillieLSTM', 'BillieTransformer']
    
    for model in models:
        mse_values = [RESULTS[size][model]['mse'] for size in dataset_sizes]
        ax.plot(dataset_sizes, mse_values, marker='o', linewidth=2.5, 
                markersize=10, label=model, color=COLORS[model])
        
        # Annotazioni
        for size, mse in zip(dataset_sizes, mse_values):
            ax.annotate(f'{mse:.5f}', 
                       xy=(size, mse), 
                       xytext=(5, 5), 
                       textcoords='offset points',
                       fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS[model], alpha=0.3))
    
    ax.set_xlabel('Dataset Size (numero di canzoni)', fontweight='bold', fontsize=12)
    ax.set_ylabel('MSE (Mean Squared Error)', fontweight='bold', fontsize=12)
    ax.set_title('Evoluzione MSE all\'Aumentare del Dataset\n(Trend di Apprendimento)', 
                 fontweight='bold', fontsize=14, pad=20)
    ax.set_xticks(dataset_sizes)
    ax.set_xticklabels([f'{size}' for size in dataset_sizes])
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# GRAFICO 4: Efficienza (Parametri vs Performance)
# ═══════════════════════════════════════════════════════════════════════════

def plot_efficiency():
    """
    Scatter plot: Numero parametri vs MSE (efficienza del modello).
    """
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    models = ['BillieMLP', 'BillieLSTM', 'BillieTransformer']
    
    for size in [50, 250, 500]:
        for model in models:
            params = RESULTS[size][model]['params']
            mse = RESULTS[size][model]['mse']
            
            ax.scatter(params, mse, s=200, alpha=0.6, 
                      color=COLORS[model], edgecolors='black', linewidth=1.5,
                      label=f'{model} ({size})' if size == 50 else "")
            
            # Annotazioni
            if size == 500:  # Annota solo dataset più grande per chiarezza
                ax.annotate(f'{model}\n{size} canzoni', 
                           xy=(params, mse),
                           xytext=(10, -10), 
                           textcoords='offset points',
                           fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.5', 
                                   facecolor=COLORS[model], alpha=0.3),
                           arrowprops=dict(arrowstyle='->', lw=1))
    
    ax.set_xlabel('Numero di Parametri', fontweight='bold', fontsize=12)
    ax.set_ylabel('MSE (Mean Squared Error)', fontweight='bold', fontsize=12)
    ax.set_title('Efficienza: Complessità del Modello vs Performance\n(In basso a sinistra = Più efficiente)', 
                 fontweight='bold', fontsize=14, pad=20)
    
    # Legend personalizzata (solo modelli, non dataset)
    handles = [mpatches.Patch(color=COLORS[m], label=m) for m in models]
    ax.legend(handles=handles, loc='upper right', framealpha=0.9)
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# GRAFICO 5: Epoche di Convergenza
# ═══════════════════════════════════════════════════════════════════════════

def plot_convergence_epochs():
    """
    Grafico a barre: Numero di epoche necessarie per convergenza.
    """
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    dataset_sizes = [50, 250, 500]
    models = ['BillieMLP', 'BillieLSTM', 'BillieTransformer']
    
    x = np.arange(len(dataset_sizes))
    width = 0.25
    
    for i, model in enumerate(models):
        epochs_values = [RESULTS[size][model]['epochs'] for size in dataset_sizes]
        offset = width * (i - 1)
        bars = ax.bar(x + offset, epochs_values, width, 
                      label=model, color=COLORS[model], alpha=0.8, edgecolor='black')
        
        # Aggiungi valori sopra le barre
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Dataset Size (numero di canzoni)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Epoche fino a Early Stopping', fontweight='bold', fontsize=12)
    ax.set_title('Velocità di Convergenza dei Modelli\n(Meno epoche = Convergenza più rapida)', 
                 fontweight='bold', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{size} canzoni' for size in dataset_sizes])
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# TABELLA RIASSUNTIVA
# ═══════════════════════════════════════════════════════════════════════════

def create_summary_table():
    """
    Crea DataFrame riassuntivo per export/stampa.
    """
    
    rows = []
    
    for size in [50, 250, 500]:
        for model in ['BillieMLP', 'BillieLSTM', 'BillieTransformer']:
            data = RESULTS[size][model]
            rows.append({
                'Dataset Size': size,
                'Modello': model,
                'MSE': f"{data['mse']:.6f}",
                'Cosine Similarity': f"{data['cosine']:.4f}",
                'Parametri': f"{data['params']:,}",
                'Epoche': data['epochs']
            })
    
    df = pd.DataFrame(rows)
    
    print("\n" + "="*80)
    print("TABELLA RIASSUNTIVA COMPLETA")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    
    return df


# ═══════════════════════════════════════════════════════════════════════════
# ANALISI VINCITORI
# ═══════════════════════════════════════════════════════════════════════════

def print_winners_analysis():
    """
    Stampa analisi dettagliata dei vincitori per categoria.
    """
    
    print("\n" + "🏆"*40)
    print("ANALISI VINCITORI PER CATEGORIA")
    print("🏆"*40 + "\n")
    
    # Vincitori per MSE
    print("📊 MIGLIOR MSE (più basso = migliore):")
    print("-" * 60)
    for size in [50, 250, 500]:
        best_model = min(['BillieMLP', 'BillieLSTM', 'BillieTransformer'], 
                        key=lambda m: RESULTS[size][m]['mse'])
        best_mse = RESULTS[size][best_model]['mse']
        print(f"  Dataset {size:3d} canzoni: {best_model:20} → MSE = {best_mse:.6f}")
    
    # Vincitori per Cosine
    print("\n📐 MIGLIOR COSINE SIMILARITY (più alto = migliore):")
    print("-" * 60)
    for size in [50, 250, 500]:
        best_model = max(['BillieMLP', 'BillieLSTM', 'BillieTransformer'], 
                        key=lambda m: RESULTS[size][m]['cosine'])
        best_cosine = RESULTS[size][best_model]['cosine']
        print(f"  Dataset {size:3d} canzoni: {best_model:20} → Cosine = {best_cosine:.4f}")
    
    # Modello più efficiente (parametri/performance)
    print("\n⚡ MODELLO PIÙ EFFICIENTE (meno parametri, buona performance):")
    print("-" * 60)
    for size in [50, 250, 500]:
        # Efficienza = performance / parametri
        efficiencies = {
            model: (1 / RESULTS[size][model]['mse']) / RESULTS[size][model]['params']
            for model in ['BillieMLP', 'BillieLSTM', 'BillieTransformer']
        }
        best_efficient = max(efficiencies, key=efficiencies.get)
        print(f"  Dataset {size:3d} canzoni: {best_efficient:20} → Migliore rapporto qualità/complessità")
    
    # Convergenza più rapida
    print("\n🚀 CONVERGENZA PIÙ RAPIDA (meno epoche):")
    print("-" * 60)
    for size in [50, 250, 500]:
        fastest = min(['BillieMLP', 'BillieLSTM', 'BillieTransformer'], 
                     key=lambda m: RESULTS[size][m]['epochs'])
        epochs = RESULTS[size][fastest]['epochs']
        print(f"  Dataset {size:3d} canzoni: {fastest:20} → {epochs} epoche")
    
    print("\n" + "="*60 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN: Genera Tutti i Grafici
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """
    Genera tutti i grafici e le analisi.
    """
    
    print("\n" + "="*80)
    print("🎨 GENERAZIONE VISUALIZZAZIONI - BILLIE AI-LISH")
    print("="*80 + "\n")
    
    # Output directory
    output_dir = Path("../data/visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Genera grafici
    print("📊 Generazione grafici in corso...")
    
    figures = {
        'mse_comparison.png': plot_mse_comparison(),
        'cosine_comparison.png': plot_cosine_comparison(),
        'mse_trends.png': plot_mse_trends(),
        'efficiency.png': plot_efficiency(),
        'convergence_epochs.png': plot_convergence_epochs()
    }
    
    # Salva grafici
    for filename, fig in figures.items():
        filepath = output_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  ✅ Salvato: {filepath}")
        plt.close(fig)
    
    # Tabella riassuntiva
    print("\n📋 Creazione tabella riassuntiva...")
    df = create_summary_table()
    
    # Salva CSV
    csv_path = output_dir / 'results_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"  ✅ Tabella salvata: {csv_path}")
    
    # Analisi vincitori
    print_winners_analysis()
    
    print("="*80)
    print("✅ TUTTE LE VISUALIZZAZIONI GENERATE CON SUCCESSO")
    print(f"📁 Location: {output_dir.absolute()}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()