# Correzioni e Razionale del Progetto

Questo documento riassume le correzioni fatte durante la revisione end-to-end del progetto `wesad-wrist-stress-prediction`, dal setup iniziale fino al full check su dati WESAD reali con GPU.

## 1. Recupero e lettura della repo

### Cosa e stato fatto

- Clonata la repo corretta `Krim0k27/wesad-wrist-stress-prediction`.
- Verificato branch e remote Git.
- Letti i PDF in `documents/`, in particolare:
  - istruzioni del final project;
  - checklist journal club;
  - paper WESAD;
  - criteri di valutazione.

### Perche

All'inizio la cartella corrente era collegata a un'altra repo (`HealthCare`). Serviva lavorare sulla repo giusta e capire i vincoli ufficiali del progetto.

Dai PDF sono emersi vincoli fondamentali:

- usare solo segnali da polso Empatica E4;
- scartare i segnali chest/RespiBAN;
- usare leave-one-subject-out cross-validation;
- usare modelli diversi dal paper originale;
- discutere bias, generalizzabilita, explainability, etica e limiti.

## 2. Valutazione critica iniziale

### Cosa e stato identificato

La prima revisione critica ha individuato vari rischi metodologici:

- il task multiclass era meno solido del task binario;
- la classe `Amusement` era debole e difficile da separare;
- il dataset WESAD ha solo 15 soggetti, quindi la generalizzazione e limitata;
- le finestre molto sovrapposte possono dare una falsa impressione di tanti campioni indipendenti;
- il binary evaluation iniziale era derivato dalle predizioni multiclass, non da un vero modello binario;
- l'XAI poteva essere calcolata su modelli allenati su tutti i dati, quindi con rischio di leakage interpretativo;
- il report del best model dipendeva dall'ordine dei modelli, non da una metrica robusta.

### Perche

Questi punti avrebbero reso il progetto fragile in sede di discussione orale. Il rischio principale era presentare il sistema come stress detection real-world, mentre il dataset misura condizioni sperimentali WESAD in laboratorio.

## 3. Metriche rese piu robuste

### Cosa e stato cambiato

In `src/models.py` e `src/evaluation.py` sono state aggiunte metriche piu adatte a classi sbilanciate:

- `Balanced Accuracy`;
- `Macro Precision`;
- `Macro Recall`;
- `Macro F1`;
- mantenimento delle metriche weighted per confronto.

Il best model ora viene scelto usando `Macro F1`, non l'ordine del dizionario.

### Perche

Le metriche weighted possono nascondere il cattivo comportamento sulle classi minoritarie. In WESAD la distribuzione delle classi non e bilanciata:

- baseline e maggioritaria;
- stress e intermedia;
- amusement e minoritaria.

`Macro F1` pesa ogni classe allo stesso modo ed e piu onesta per il report.

## 4. Vero task binario stress vs non-stress

### Cosa e stato cambiato

In `scripts/train_evaluate.py` e stato aggiunto un vero training binario LOSO:

- `stress = 1`;
- `baseline + amusement = 0`;
- training separato dal multiclass;
- metriche binarie dedicate;
- salvataggio in `metrics_binary.csv` e `metrics_binary.json`.

Le vecchie metriche binarie derivate dal multiclass vengono salvate solo come diagnostica in:

```text
metrics_binary_from_multiclass.csv
```

### Perche

Collassare le predizioni multiclass in binario non equivale ad allenare un classificatore binario. Per il progetto, il task piu difendibile e:

```text
stress vs non-stress
```

Questo task e piu coerente con l'obiettivo reale e produce risultati piu solidi.

## 5. Windowing anti-leakage alle transizioni

### Cosa e stato cambiato

In `src/preprocessing.py` e `scripts/build_dataset.py` e stato aggiunto il controllo:

```text
require_pure_physio_window = true
```

Di default una finestra fisiologica da 60 secondi viene tenuta solo se tutta la finestra appartiene alla stessa label.

E stato anche aggiunto:

```text
transition_buffer_sec
```

per poter scartare un buffer aggiuntivo vicino alle transizioni.

### Perche

La pipeline originale assegnava la label usando il tempo finale della finestra. Questo puo creare finestre problematiche vicino alle transizioni, per esempio una finestra etichettata come `stress` ma contenente parte della baseline precedente.

La correzione riduce leakage temporale e rende la validazione piu difendibile.

## 6. Supporto a struttura raw WESAD annidata

### Cosa e stato cambiato

In `src/loader.py` e stata aggiunta la funzione:

```python
resolve_raw_dir(...)
```

La pipeline ora riconosce sia:

```text
data/01_raw/WESAD/S2/...
```

sia:

```text
data/01_raw/WESAD/WESAD/S2/...
```

### Perche

Il dataset scaricato era stato estratto con una cartella `WESAD` annidata. Senza questa correzione, `build_dataset.py` non trovava i soggetti.

La modifica rende il setup meno fragile.

## 7. Correzione feature BVP e ACC

### Cosa e stato trovato

Durante il sanity check del dataset processato sono emersi problemi numerici:

- `bvp_hrv_lf_hf_ratio` aveva outlier enormi, fino a circa `1e32`;
- `acc_mag_peak_freq` era costante a zero;
- alcune peak frequency ACC erano quasi sempre zero.

### Cosa e stato cambiato

In `src/features.py`:

- per le feature ACC in frequenza viene ignorata la componente DC dell'FFT;
- per BVP vengono filtrati intervalli con heart rate non plausibile;
- `LF/HF ratio` viene calcolato solo se `HF` e sufficientemente grande;
- il ratio viene limitato per evitare esplosioni numeriche.

### Perche

Outlier numerici enormi possono dominare feature selection, scaling e training. La componente DC nell'FFT faceva selezionare frequenza zero come picco, che non e informativa per il movimento.

Queste correzioni rendono le feature piu stabili e interpretabili.

## 8. Feature selection piu sicura

### Cosa e stato cambiato

In `scripts/train_evaluate.py` la feature pipeline ora include:

```python
variance_threshold = 1e-12
```

Sono stati aggiunti anche parametri CLI:

```bash
--scaler
--selection-k
--selection-method
```

### Perche

Il dataset aveva almeno una feature costante (`bvp_hrv_ulf`). Le feature costanti non aiutano il modello e possono disturbare alcune analisi.

I parametri CLI rendono gli esperimenti piu espliciti e riproducibili.

## 9. XAI resa piu difendibile

### Cosa e stato cambiato

In `src/explainability.py` e stata aggiunta:

```python
compute_loso_permutation_importance(...)
```

Questa funzione calcola permutation importance su soggetti held-out fold by fold.

### Perche

Calcolare l'importanza delle feature su un modello allenato su tutti i dati puo essere fuorviante. L'interpretazione deve riflettere il comportamento su soggetti non visti, coerentemente con LOSO.

## 10. Backend Matplotlib headless

### Cosa e stato cambiato

In `src/explainability.py` e stato forzato:

```python
matplotlib.use("Agg", force=True)
```

### Perche

Durante uno smoke test Matplotlib provava a usare Tk, ma l'installazione locale non aveva Tcl completo. Il backend `Agg` permette di generare PNG senza GUI, utile su Windows, server o ambienti headless.

## 11. Modelli class-balanced

### Cosa e stato cambiato

In `src/models.py` sono stati aggiunti:

- `SVM Balanced`;
- `Logistic Regression Balanced`;
- `Torch MLP Balanced`.

### Perche

WESAD e sbilanciato tra classi. I modelli class-balanced aiutano a non sacrificare le classi minoritarie, soprattutto `Amusement` nel multiclass e `Stress` nel binario.

## 12. Supporto GPU reale con PyTorch

### Cosa e stato cambiato

E stato installato PyTorch CUDA:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

In `src/models.py` e stato aggiunto:

```python
TorchMLPClassifier
```

che e compatibile con la pipeline sklearn-like e usa CUDA quando disponibile.

Sono stati aggiunti anche i flag:

```bash
--models
--torch-epochs
--torch-batch-size
```

### Perche

La GPU era presente ma Python non aveva backend GPU installati:

```text
torch=False
cuml=False
cupy=False
```

Inoltre cuML/RAPIDS su Windows non e una strada pratica nella maggior parte dei casi. PyTorch CUDA era il modo piu realistico per usare davvero la RTX 2080 SUPER.

## 13. Full check sui dati raw

### Cosa e stato controllato

Sono stati verificati tutti i 15 soggetti validi:

```text
S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S13, S14, S15, S16, S17
```

Per ogni soggetto sono stati controllati:

- presenza dei file `.pkl`;
- segnali wrist `ACC`, `BVP`, `EDA`, `TEMP`;
- durata allineata tra segnali e label;
- presenza delle label `1`, `2`, `3`;
- assenza di segnali chest nella rappresentazione caricata.

### Risultato

Tutti i soggetti sono coerenti. Nessun soggetto raw problematico.

## 14. Dataset processato rigenerato

### Cosa e stato prodotto

E stato rigenerato:

```text
data/03_processed/features.npz
```

con:

- `121853` finestre;
- `51` feature;
- `15` soggetti;
- nessun `NaN`;
- nessun `inf`.

Distribuzione classi:

```text
Baseline: 66859
Stress: 36279
Amusement: 18715
```

### Perche

Serviva ricostruire il dataset dopo le correzioni a windowing e feature extraction.

## 15. Full evaluation GPU

### Comando usato

```bash
python scripts/train_evaluate.py --backend gpu --skip-xai --skip-lstm --models "Torch MLP,Torch MLP Balanced"
```

### Hardware rilevato

```text
GPU: NVIDIA GeForce RTX 2080 SUPER
CUDA disponibile: true
PyTorch: 2.11.0+cu128
```

### Tempo

Circa 50 minuti per:

- 15 fold LOSO multiclass;
- 15 fold LOSO binario;
- 2 modelli Torch;
- 35 epoche per training.

Il tempo e alto perche il run esegue molti training indipendenti:

```text
15 fold x 2 modelli x 2 task = 60 training
```

## 16. Risultati principali

### Multiclass

Best model:

```text
Torch MLP Balanced
```

Metriche:

```text
Accuracy:          0.677
Balanced Accuracy: 0.610
Macro F1:          0.609
Weighted F1:       0.684
```

Report classi:

```text
Baseline   F1 = 0.75
Stress     F1 = 0.75
Amusement  F1 = 0.33
```

Interpretazione:

Il multiclass funziona discretamente per baseline e stress, ma resta debole su amusement. Questo va dichiarato nel report.

### Binary stress vs non-stress

Best model:

```text
Torch MLP
```

Metriche:

```text
Accuracy:          0.845
Balanced Accuracy: 0.799
Macro F1:          0.808
Weighted F1:       0.842
Stress F1:         0.724
Stress recall:     0.683
Stress precision:  0.771
Specificity:       0.914
```

Interpretazione:

Il task binario e molto piu solido e dovrebbe essere presentato come risultato principale.

## 17. Output puliti

Per evitare confusione con artefatti vecchi gia presenti in `experiments/run_001`, e stata creata una cartella pulita:

```text
experiments/run_gpu_torch
```

Contiene:

- metriche multiclass;
- metriche binarie;
- classification report;
- confusion matrix;
- per-subject metrics;
- subject error analysis;
- risultati LOSO salvati.

## 18. Documentazione aggiornata

Sono stati aggiornati:

- `README.md`;
- `TODO.md`;
- `config.yaml`.

### Perche

Per rendere il workflow riproducibile e indicare chiaramente:

- come installare PyTorch CUDA;
- come lanciare il run GPU;
- quanto tempo aspettarsi;
- come fare uno smoke test veloce con meno epoche.

## 19. Raccomandazione finale per il report

La versione piu difendibile del progetto e:

```text
WESAD wrist-only stress vs non-stress classification with LOSO validation.
```

Il multiclass va incluso come analisi secondaria, spiegando che `Amusement` e difficile da distinguere per:

- classe piu piccola;
- risposta fisiologica piu debole;
- somiglianza parziale con baseline;
- variabilita individuale.

Nel report conviene enfatizzare:

- uso esclusivo dei segnali da polso;
- LOSO subject-independent;
- task binario principale;
- limiti di WESAD;
- rischio di generalizzazione limitata;
- explainability e analisi per soggetto;
- impossibilita di dichiarare validita clinica o real-world senza validazione esterna.

## 20. Comandi utili

Build dataset:

```bash
python scripts/build_dataset.py
```

Run GPU completo:

```bash
python scripts/train_evaluate.py --backend gpu --skip-xai --skip-lstm --models "Torch MLP,Torch MLP Balanced"
```

Run GPU veloce:

```bash
python scripts/train_evaluate.py --backend gpu --skip-xai --skip-lstm --models "Torch MLP,Torch MLP Balanced" --torch-epochs 5
```

Run con XAI fold-wise:

```bash
python scripts/train_evaluate.py --backend gpu --skip-lstm --models "Torch MLP,Torch MLP Balanced"
```

Nota: il run con XAI richiede piu tempo perche calcola permutation importance sui fold held-out.
