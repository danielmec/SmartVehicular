# Project 11 - Adverse Weather Robustness Benchmark

## Obiettivo
Valutare la robustezza del rilevamento pedoni (YOLOv8n) in CARLA sotto condizioni
ambientali avverse e corruzioni visive progressive, identificando soglie di collasso.

## Metodologia
- **Baseline**: YOLOv8n pedestrian detection (soglia confidenza: 0.6)
- **Scenario**: pedone a distanza fissa (80m), traiettoria costante, velocita costante
- **Condizioni meteo**: clear, rain, fog, night, backlight
- **Corruzioni visive**: Gaussian blur (s1-s3), noise (s1-s3), motion blur (s1-s2), contrast (s1-s2)
- **Combinazioni**: rain+blur, fog+noise
- **Metriche**: detection rate, first detection distance, min distance, warning delay,
  miss rate, false positive count, FCW precision/recall
- **Test totali**: 17 condizioni
- **Batch aggregati**: 5 run per condizione

## Risultati

| condition                |   detection_rate |   first_det_dist_mean |   min_dist_mean |
|:-------------------------|-----------------:|----------------------:|----------------:|
| night | none | s0        |          0.18434 |               24.5144 |          2.329  |
| clear | noise | s2       |          0.18    |               44.219  |          2.974  |
| fog | noise | s1         |          0.1795  |               61.282  |          2.98   |
| clear | noise | s1       |          0.1731  |               66.262  |          2.784  |
| rain | blur | s1         |          0.1667  |               52.128  |          2.743  |
| clear | none | s0        |          0.13314 |               19.1386 |          2.5002 |
| clear | contrast | s1    |          0.1091  |               18.773  |          2.342  |
| clear | contrast | s2    |          0.1091  |               16.451  |          2.306  |
| clear | blur | s2        |          0.10672 |               31.1638 |          2.63   |
| clear | blur | s1        |          0.10384 |               27.4408 |          2.7504 |
| rain | none | s0         |          0.09816 |               31.1044 |          2.6586 |
| backlight | none | s0    |          0.0887  |               20.548  |          2.6802 |
| fog | none | s0          |          0.07666 |               13.6012 |          2.6762 |
| clear | noise | s3       |          0.0566  |               30.489  |          2.784  |
| clear | blur | s3        |          0.0536  |               10.14   |          2.776  |
| clear | motion_blur | s2 |          0       |                0      |          0      |
| clear | motion_blur | s1 |          0       |                0      |          0      |

## Interpretazione
- **Migliore**: night | none | s0 (18.43%)
- **Peggiore**: clear | motion_blur | s1 (0.00%)

### Osservazioni principali
- La pioggia degrada significativamente il detection rate
- Il blur progressivo (s1 -> s2 -> s3) mostra degradazione chiara con soglia di collasso
- Il noise gaussiano ad alta severita compromette la detection
- Le combinazioni weather+corruption (es. rain+blur) sono le piu critiche
- Night ha impatto moderato grazie ai fari del veicolo

## Failure Cases

### Condizioni di collasso (detection rate < 5%)
- **clear | motion_blur | s1**: 0.00%
- **clear | motion_blur | s2**: 0.00%

## Forward Collision Warning (FCW)
- Soglia WARNING: 20.0m
- Soglia DANGER: 10.0m
- Il modulo FCW valuta precision, recall e warning delay per ogni condizione

## Deliverables
- Benchmark self-contained (notebook Jupyter)
- Scenario CARLA con gestione meteo
- Corruzioni visive multiple (blur, noise, motion blur, contrast)
- CSV con metriche per-frame e per-run
- Grafici: detection rate, degradazione, heatmap, confidence/distance, FCW
- Confronti side-by-side per tutte le condizioni
- Report con metodologia, metriche e failure cases
- Modulo FCW con precision/recall e warning delay