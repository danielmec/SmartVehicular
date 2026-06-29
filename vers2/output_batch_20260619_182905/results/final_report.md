# Project 11 — Adverse Weather Robustness Benchmark

## Obiettivo
Valutare la robustezza del rilevamento pedoni (YOLOv8n) in CARLA sotto condizioni ambientali avverse.

## Metodologia
- **Baseline**: YOLOv8n pedestrian detection
- **Scenario**: pedone a distanza fissa, traiettoria costante
- **Condizioni**: clear, rain, fog, night, backlight + blur (s1, s2)
- **Metriche**: detection rate, first detection distance, min distance

## Risultati Batch 20260619_182924

| condition             |   detection_rate |   first_detection_distance_m |   min_distance_m | status   |
|:----------------------|-----------------:|-----------------------------:|-----------------:|:---------|
| clear | none | s0     |           0.1111 |                       17.137 |            2.373 | ok       |
| rain | none | s0      |           0.0179 |                        2.13  |            2.13  | ok       |
| fog | none | s0       |           0.0909 |                       13.764 |            2.733 | ok       |
| night | none | s0     |           0.1404 |                       17.198 |            2.114 | ok       |
| backlight | none | s0 |           0.0889 |                       13.799 |            2.657 | ok       |
| clear | blur | s1     |           0.1034 |                       15.832 |            2.658 | ok       |
| clear | blur | s2     |           0.0339 |                        6.544 |            2.742 | ok       |

## Interpretazione
- **Migliore**: night | none | s0 (14.04%)
- **Peggiore**: rain | none | s0 (1.79%)

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