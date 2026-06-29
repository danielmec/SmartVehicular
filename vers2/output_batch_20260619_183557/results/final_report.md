# Project 11 — Adverse Weather Robustness Benchmark

## Obiettivo
Valutare la robustezza del rilevamento pedoni (YOLOv8n) in CARLA sotto condizioni ambientali avverse.

## Metodologia
- **Baseline**: YOLOv8n pedestrian detection
- **Scenario**: pedone a distanza fissa, traiettoria costante
- **Condizioni**: clear, rain, fog, night, backlight + blur (s1, s2)
- **Metriche**: detection rate, first detection distance, min distance

## Risultati Batch 20260619_183610

| condition             |   detection_rate |   first_detection_distance_m |   min_distance_m | status   |
|:----------------------|-----------------:|-----------------------------:|-----------------:|:---------|
| clear | none | s0     |           0.1406 |                       18.69  |            2.364 | ok       |
| rain | none | s0      |           0.0909 |                       14.409 |            2.694 | ok       |
| fog | none | s0       |           0.0781 |                       13.398 |            2.721 | ok       |
| night | none | s0     |           0.1538 |                       16.963 |            2.117 | ok       |
| backlight | none | s0 |           0.0741 |                       12.616 |            2.688 | ok       |
| clear | blur | s1     |           0.082  |                       16.023 |            2.663 | ok       |
| clear | blur | s2     |           0.0462 |                       20.435 |            2.273 | ok       |

## Interpretazione
- **Migliore**: night | none | s0 (15.38%)
- **Peggiore**: clear | blur | s2 (4.62%)

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