

# Billie AI-lish 

**Billie AI-lish** è un sistema di raccomandazione musicale sequenziale basato su **Deep Learning**. A differenza dei classici algoritmi di raccomandazione, questo sistema utilizza reti neurali per analizzare l'ordine temporale degli ascolti e prevedere quale sarà la "prossima canzone ideale" per l'utente, mantenendo la coerenza del mood e del flow musicale.

## Obiettivo del Progetto

L'obiettivo è confrontare diverse architetture di reti neurali (**MLP**, **LSTM**, **Transformer**) per capire quale sia la più efficace nel modellare i gusti musicali di un utente basandosi su uno storico di ascolti limitato (Small/Medium Data).

## Struttura delle Cartelle

Il progetto è organizzato per separare la logica di acquisizione dati da quella di sperimentazione:

* **`src/`**: Contiene gli script originali e una versione precedente dell'applicazione. È qui che si trova la logica per il **fetch della user_history** tramite API di Spotify.
* **`model-comparison/`**: Il nucleo della ricerca attuale. Contiene gli script per il benchmark dei modelli e la nuova dashboard interattiva.

## Architettura del Codice

Il progetto è suddiviso in moduli logici per garantire manutenibilità e rigore scientifico:

* **`architectures.py`**: Definizioni PyTorch per MLP (statico), LSTM (ricorrente) e Transformer (self-attention).
* **`data_factory.py`**: Gestione del caricamento dati, normalizzazione delle 9 feature audio e split sequenziale pulito (senza overlap).
* **`run_experiment.py`**: Motore di training che esegue il benchmark su dataset di taglia 50, 250 e 500. Include riproducibilità totale tramite seed e ottimizzazione con LR Scheduler.
* **`lstm_recommender.py`**: Modulo di inferenza che trasforma la predizione del modello in raccomandazioni reali tramite *Cosine Similarity* con il database brani.
* **`streamlit_app.py`**: Interfaccia grafica sobria per generare playlist e analizzare i vettori di predizione.

## Come avviare il progetto

### 1. Requisiti e Credenziali

Assicurati di avere Python installato e le dipendenze necessarie:

```bash
pip install torch pandas numpy scikit-learn streamlit plotly python-dotenv

```

Per scaricare i tuoi dati da Spotify, devi creare un file **`.env`** nella root del progetto con le tue credenziali da sviluppatore (ottenibili sulla [Spotify Developer Dashboard](https://developer.spotify.com/)):

```env
SPOTIPY_CLIENT_ID='il_tuo_client_id'
SPOTIPY_CLIENT_SECRET='il_tuo_client_secret'
SPOTIPY_REDIRECT_URI='http://localhost:8888/callback'

```

### 2. Preparazione Dati

Inserisci (o genera tramite gli script in `src/`) i dati nella cartella `data/`:

* `user_history.csv`: Lo storico dei tuoi ascolti.
* `tracks_processed.csv`: Il database globale delle canzoni (2.8M+ tracks).

### 3. Addestramento dei Modelli

Esegui il benchmark per addestrare le IA e generare i dati per la dashboard:

```bash
python model-comparison/run-experiment.py

```

I modelli migliori verranno salvati in `data/trained_models/`.

### 4. Avvio della Dashboard

Per interagire con il sistema e generare la tua playlist:

```bash
streamlit run model-comparison/streamlit_app.py

```

---

**Nota Tecnica**: Il sistema utilizza 9 feature audio fondamentali (Energy, Valence, Danceability, ecc.) normalizzate tra 0 e 1. Dagli esperimenti è emerso che l'architettura **LSTM** risulta la più efficace nel bilanciare precisione e capacità di generalizzazione su dataset di medie dimensioni.

---

