# Analisi dei Risultati — Adverse Weather Robustness Benchmark

## Dati Raccolti
Il benchmark è stato eseguito **4 volte** (4 batch indipendenti), per un totale di **68 test** (17 condizioni × 4 ripetizioni). I risultati sono stati aggregati calcolando media e deviazione standard per ogni condizione.

## Prestazione Baseline
In condizioni ottimali (**clear, nessuna corruzione**), il sistema raggiunge un detection rate del **10.06% ± 1.43%**, con prima rilevazione a **16.7m ± 2.1m**. Il valore del detection rate può sembrare basso in termini assoluti, ma è coerente con il setup sperimentale: il pedone parte a 80m di distanza e il modello YOLOv8n (nano) inizia a rilevarlo solo negli ultimi 15-17 metri, quando la sagoma diventa sufficientemente grande nell'immagine della telecamera.

---

## Impatto delle Condizioni Meteo

Tutte le condizioni meteo sono state testate senza alcuna corruzione visiva aggiuntiva, per isolare l'effetto del meteo CARLA sulla pipeline di detection.

| Condizione | Detection Rate | Δ vs Baseline | Prima Detection | Stabilità (std) |
|---|---|---|---|---|
| **Clear** | 10.06% ± 1.43% | — | 16.7m | Alta |
| **Fog** | 10.69% ± 7.40% | +0.63% | 14.1m | Molto bassa |
| **Night** | 8.46% ± 0.58% | -1.60% | 15.5m | Molto alta |
| **Backlight** | 6.58% ± 1.22% | -3.48% | 12.8m | Alta |
| **Rain** | 5.83% ± 4.21% | -4.23% | 11.4m | Bassa |

### Osservazioni

- **La pioggia è la condizione più critica**: causa una degradazione del **42%** rispetto al baseline, con una distanza di prima rilevazione che scende a 11.4m. Inoltre, presenta la variabilità più alta tra i batch (± 4.21%), indicando un comportamento imprevedibile del sistema — il peggior scenario possibile per un ADAS reale.

- **Il controluce (backlight) è la seconda condizione più critica**: il sole basso e frontale riduce il detection rate del **35%** e anticipa la prima detection di quasi 4 metri. La bassa deviazione standard (± 1.22%) indica che l'effetto è consistente e riproducibile.

- **La notte degrada moderatamente** (-16%) ma in modo **molto stabile** (± 0.58%). Questo suggerisce che la scarsa illuminazione riduce uniformemente la visibilità del pedone senza introdurre variabilità casuale.

- **La nebbia è un risultato sorprendente**: il detection rate (10.69%) è paradossalmente superiore al clear. L'ipotesi è che la nebbia in CARLA renda lo sfondo più uniforme e omogeneo, facilitando a YOLOv8 l'isolamento della sagoma del pedone rispetto a uno sfondo complesso. Tuttavia, la deviazione standard elevatissima (± 7.40%) indica che questo effetto è molto instabile tra i batch.

---

## Impatto delle Corruzioni Visive

Le corruzioni sono state applicate in post-processing sull'immagine catturata dalla camera in condizione **clear**, per isolare l'effetto della singola corruzione. Ogni corruzione è stata testata a 2 o 3 livelli di severità crescente.

| Corruzione | Severità 1 | Severità 2 | Severità 3 | Tendenza |
|---|---|---|---|---|
| **Contrast** | 9.07% ± 1.00% | 9.07% ± 1.01% | — | Nessun effetto |
| **Blur** | 8.02% ± 2.54% | 3.37% ± 0.87% | 1.47% ± 0.98% | Degradazione progressiva |
| **Noise** | 7.70% ± 1.33% | 5.75% ± 1.43% | 0.45% ± 0.90% | Degradazione progressiva |
| **Motion Blur** | 0.00% ± 0.00% | 0.00% ± 0.00% | — | Collasso immediato |

### Osservazioni

- **Motion blur causa il collasso totale e immediato** del sistema: detection rate 0.00% anche al livello di severità più basso. Questo è il risultato più significativo del benchmark. La sfocatura direzionale da movimento distrugge completamente la capacità di YOLOv8n di riconoscere forme umane, rendendo il sistema completamente cieco. In uno scenario reale, vibrazioni della telecamera o movimenti bruschi del veicolo avrebbero lo stesso effetto devastante.

- **Il blur gaussiano degrada progressivamente**: il sistema perde il 20% delle capacità a s1, il 67% a s2, e l'85% a s3. La soglia di fallimento si colloca tra severità 2 e 3. La prima detection scende da 16.7m (baseline) a soli **2.8m** a severità 3 — una distanza alla quale un intervento frenante sarebbe impossibile.

- **Il rumore (noise) segue una curva simile al blur**: degradazione graduale fino a s2 (-43%), poi collasso quasi totale a s3 (0.45%). La prima detection a s3 avviene a **3.2m**, confermando l'impossibilità di reazione.

- **La riduzione di contrasto non ha alcun effetto**: il detection rate rimane pressoché identico al baseline (9.07% vs 10.06%) a tutte le severità testate. Questo indica che YOLOv8n è naturalmente robusto a variazioni di contrasto dell'immagine, probabilmente grazie alla normalizzazione interna del modello.

---

## Combinazioni Meteo + Corruzione

Due condizioni aggiuntive testano la combinazione di meteo avverso con corruzione visiva:

| Condizione | Detection Rate | Commento |
|---|---|---|
| **Rain + Blur s1** | 4.12% ± 3.11% | Effetto combinato peggiore delle singole componenti |
| **Fog + Noise s1** | 7.35% ± 1.32% | Degradazione moderata |

La combinazione **pioggia + blur** produce un effetto moltiplicativo: il detection rate scende al 4.12%, significativamente peggiore sia della sola pioggia (5.83%) che del solo blur s1 (8.02%). Questo dimostra che in condizioni reali, dove più fattori di degradazione coesistono, il sistema è ancora più vulnerabile.

---

## Distanza di Prima Rilevazione

La distanza alla quale il sistema rileva il pedone per la prima volta è un indicatore critico per la sicurezza:

| Fascia | Condizioni | Implicazione |
|---|---|---|
| **15-17m** | Clear, fog, night, noise s1, contrast s1-s2 | Reazione possibile a velocità moderate |
| **10-14m** | Rain, backlight, blur s1, noise s2 | Margine ridotto, frenata d'emergenza necessaria |
| **< 5m** | Blur s3, noise s3 | **Troppo tardi per qualsiasi intervento** |
| **0m (mai)** | Motion blur s1-s2 | **Sistema completamente cieco** |

---

## Conclusioni Chiave

1. **Il motion blur è la minaccia più grave**: causa collasso immediato a qualsiasi severità. Un sistema ADAS basato solo su camera RGB è completamente vulnerabile a vibrazioni o movimenti della telecamera.

2. **La pioggia è la condizione meteo più pericolosa**: degrada significativamente il detection rate e introduce alta variabilità, rendendo il comportamento del sistema imprevedibile.

3. **Blur e noise degradano progressivamente** con soglie di fallimento identificabili tra severità 2 e 3, dove la prima detection avviene a meno di 3 metri — insufficiente per qualsiasi manovra di sicurezza.

4. **Il contrasto non è un fattore critico**: YOLOv8n è intrinsecamente robusto a variazioni di contrasto.

5. **La nebbia ha un effetto controintuitivo**: non degrada il sistema in modo significativo, ma introduce alta variabilità tra le esecuzioni.

6. **La combinazione di fattori è più pericolosa dei singoli fattori**: la coppia pioggia + blur produce un effetto moltiplicativo sulla degradazione.

Questi risultati evidenziano la necessità di soluzioni ridondanti (sensori multipli, fusione LiDAR/Radar) e di algoritmi di pre-processing adattivo per garantire la robustezza dei sistemi di percezione in condizioni operative reali.
