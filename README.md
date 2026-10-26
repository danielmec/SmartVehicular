# Project 11: Adverse Weather Robustness Benchmark

Benchmark di robustezza di una funzione ADAS solo camera in CARLA 0.9.16. Per il banchmark si usa
YOLOv8n pre-addestrato su COCO (nessun fine-tuning) con un modulo di warning basato sul
time-to-collision (TTC). Il benchmark misura come degrada in due scenari fissi e ripetibili, al variare
del meteo e con corruzioni visive dell'immagine.

- **Scenario 1 - Pedone (FCW):** l'auto procede a 30 km/h su un rettilineo urbano mentre un pedone
  scende dallo spartitraffico e attraversa la corsia. Traffico di contorno nella carreggiata opposta,
  un'auto lenta nella corsia di destra e persone sul marciapiede.
- **Scenario 2 - Semaforo (Red Light Warning):** l'auto arriva a 40 km/h a un incrocio a T con il
  semaforo bloccato sul rosso, con traffico trasversale.

Ogni condizione viene classificata come **OK**, **DEGRADATO** o **COLLASSO** rispetto alla baseline in
clear.

## Requisiti

- **CARLA 0.9.16** (server), con la mappa `Town10HD_Opt`, che è quella caricata all'avvio di CARLA.
- **Python 3.10**.
- **GPU NVIDIA con CUDA**, consigliata. Il benchmark funziona anche su CPU, con gli stessi risultati ma
  molto più lentamente.

## Creazione dell'ambiente conda

Da Anaconda Prompt:

```bat
:: 1. Crea e attiva l'ambiente (va bene qualunque 3.10.x)
conda create -n adverseWeather python=3.10.12 -y
conda activate adverseWeather

:: 2. Controlla la versione CUDA del driver: in alto a destra "CUDA Version"
nvidia-smi

:: 3. PyTorch con CUDA, PRIMA dei requirements
::    cu121 con "CUDA Version" 12.1 o superiore, cu118 con un driver 11.x
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

:: 4. Kernel per Jupyter
python -m pip install ipykernel
python -m ipykernel install --user --name adverseWeather --display-name "Project 11 - CARLA 0.9.16 (Python 3.10)"
```

Note:

- Il passo 3 va fatto prima di `requirements.txt`: su Windows `pip install ultralytics` installa
  altrimenti PyTorch solo per CPU. Se trova torch già installato, non lo sostituisce.
- `conda init` serve una sola volta, e solo se `conda activate` dà errore (per esempio in PowerShell o
  nel terminale di VS Code). Dopo `conda init` bisogna chiudere e riaprire il terminale.
- Le altre dipendenze (`carla==0.9.16`, `ultralytics`, `numpy`, `pandas`, `matplotlib`, `seaborn`, ...)
  si installano dal notebook, Sezione 1.

## Esecuzione con il notebook

1. Avviare il server CARLA con la qualità di default:
   - `CarlaUE4.exe` 
2. Aprire `adverseWeather/Project11_Adverse_Weather_Benchmark_v5.ipynb` e scegliere il kernel
   **"Project 11 - CARLA 0.9.16 (Python 3.10)"**.
3. Eseguire la Sezione 1: la cella di fix delle dipendenze e `%pip install -r requirements.txt`.
   **Poi riavviare il kernel**: serve per `matplotlib-inline 0.1.6`, altrimenti i grafici danno errore.
4. Eseguire le Sezioni 1-9 (import, connessione a `localhost:2000`, meteo, scenari, modello, ground
   truth, configurazione, modulo di warning). 
5. Eseguire la **Sezione 10** (batch dello Scenario 1) e la **Sezione 11** (batch dello Scenario 2).
   Ogni batch esegue tutte le 20 condizioni senza interventi e salva i dati in una nuova cartella
   `output_batch_<timestamp>`.
6. Per un secondo batch: rieseguire la cella della **Sezione 7** (crea una nuova cartella di output) e
   poi di nuovo le Sezioni 10 e 11.
7. Eseguire le **Sezioni 12-15** per l'analisi: leggono tutti i batch completi, ne fanno la media e
   salvano tabelle, grafici, confronti side-by-side e failure case nella cartella `plots` dell'ultimo
   batch.


**Importante:** l'analisi fa la media di tutte le cartelle `output_batch_*` complete. 
## Condizioni testate

| N. | Gruppo | Condizione |
|---|---|---|
| 1-5 | Meteo base (parametri meteo di CARLA) | clear, rain, fog, night, backlight |
| 6-10 | Meteo severo o intermedio | rain severe, fog 60, fog 75, fog severe, night severe (lampioni spenti) |
| 11-12 | Combinazioni | night highbeam (lampioni spenti, abbaglianti accesi), rain night |
| 13-16 | Blur gaussiano sull'immagine, su clear | sigma 1, 2, 3 (mild) e 7 px (severe) a 1280x720 |
| 17-18 | Gocce sulla lente, su rain | mild, severe |
| 19-20 | Abbagliamento del sole, su backlight | mild, severe |

Le corruzioni sono deterministiche (gocce in posizioni fisse, sole sempre nello stesso punto), quindi
sono identiche in ogni frame e in ogni run.

## Metriche ed esiti

Per ogni frame le detection di YOLO (confidenza almeno 0.55) vengono confrontate con la ground truth di
CARLA proiettata nell'immagine (match con IoU almeno 0.3). Per ogni condizione si calcolano recall in
zona critica, recall totale, precision, distanza della prima detection, TTC al primo warning, ritardo
del warning e se il veicolo riesce ancora a fermarsi.

| Esito | Criterio |
|---|---|
| COLLASSO | il warning arriva troppo tardi per fermarsi (o non arriva), oppure la recall in zona critica scende sotto il 50% di quella in clear |
| DEGRADATO | la recall in zona critica scende sotto l'80% del clear, oppure il warning arriva più di 0.5 s dopo rispetto al clear |
| OK | nessuno dei casi precedenti |

## Output

```
output_batch_<timestamp>/
    results/                          Scenario 1
        run_summary.csv               una riga per condizione
        frame_metrics.csv             una riga per frame
        config.json                   configurazione completa e meteo applicato (scritto a batch completato)
    detection_samples/<condizione>/   immagini annotate (YOLO e ground truth)
    scenario2_traffic_light/          stessa struttura per lo Scenario 2
    plots/                            tabelle (summary_ped/tl .csv e .md) e grafici dell'analisi
```

Un batch è considerato completo, e quindi letto dall'analisi, solo se contiene `config.json`.

## Problemi comuni

- **`CUDA (Nvidia GPU) available: False`:** PyTorch è la versione per CPU. Con l'ambiente attivo:
  ```bat
  python -m pip uninstall -y torch torchvision
  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```
  e poi riavviare il kernel.

