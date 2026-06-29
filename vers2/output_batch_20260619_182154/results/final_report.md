# Project 11 — Adverse Weather Robustness Benchmark

## Obiettivo
Valutare la robustezza del rilevamento pedoni (YOLOv8n) in CARLA sotto condizioni ambientali avverse.

## Metodologia
- **Baseline**: YOLOv8n pedestrian detection
- **Scenario**: pedone a distanza fissa, traiettoria costante
- **Condizioni**: clear, rain, fog, night, backlight + blur (s1, s2)
- **Metriche**: detection rate, first detection distance, min distance

## Risultati Batch 20260619_182225

| condition             |   detection_rate |   first_detection_distance_m |   min_distance_m | status   |
|:----------------------|-----------------:|-----------------------------:|-----------------:|:---------|
| clear | none | s0     |           0.1186 |                       15.204 |            2.348 | ok       |
| rain | none | s0      |           0.0566 |                        7.21  |            2.727 | ok       |
| fog | none | s0       |           0.0862 |                       14.167 |            2.687 | ok       |
| night | none | s0     |           0.1562 |                       17.022 |            2.039 | ok       |
| backlight | none | s0 |           0.0926 |                       14.503 |            2.508 | ok       |
| clear | blur | s1     |           0.0462 |                        8.868 |            2.972 | ok       |
| clear | blur | s2     |           0.0448 |                        7.905 |            2.379 | ok       |

## Interpretazione
- **Migliore**: night | none | s0 (15.62%)
- **Peggiore**: clear | blur | s2 (4.48%)

### Osservazioni
- La pioggia degrada significativamente il detection rate
- Il blur progressivo (s1→s2) riduce ulteriormente le prestazioni
- Night e backlight mostrano performance comparabili a clear

## Deliverables
- ✅ Benchmark scripts eseguibili
- ✅ Scenario CARLA con gestione meteo
- ✅ CSV con metriche per-frame e per-run
- ✅ Grafici e confronti side-by-side
- ✅ Report con metodologia e failure cases