"""
Scenario 2 - Semaforo (Red Light Warning): esegue il benchmark su tutte le condizioni di TESTS
(cella 34 del notebook).

Output in <batch>/scenario2_traffic_light/, dove <batch> e' una nuova vers2/output_batch_<timestamp>/
oppure la cartella data con --batch-dir (per esempio quella appena creata dallo Scenario 1,
come nel notebook dove i due scenari condividono la stessa cartella):
    results/run_summary.csv, results/frame_metrics.csv, results/config.json
    detection_samples/<run_label>/*.png

I CSV vengono aggiornati dopo ogni condizione; config.json viene scritto solo a batch
completato: l'analisi (analisi_grafici.py) considera validi solo i batch con config.json.

Uso:
    carlapy scenario2_semaforo.py
    carlapy scenario2_semaforo.py --batch-dir ../output_batch_20261019_150000
    carlapy scenario2_semaforo.py --tests 1        # solo clear (prova veloce)
    carlapy scenario2_semaforo.py --senza-camera   # solo moto e spectator, senza camera ne' YOLO

Con --senza-camera le metriche di detection sono vuote: l'output va in
vers2/test_senza_camera_<timestamp>/ e non viene letto da analisi_grafici.py.
"""

import argparse
import os
from datetime import datetime

from common import *


def main():
    argparser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argparser.add_argument('--host', default=CARLA_HOST, help=f'IP del server CARLA (default: {CARLA_HOST})')
    argparser.add_argument('-p', '--port', default=CARLA_PORT, type=int, help=f'porta TCP (default: {CARLA_PORT})')
    argparser.add_argument('--tests', metavar='N,N', help='solo alcuni test, numeri 1-based (default: tutti)')
    argparser.add_argument('--batch-dir', help='cartella di output esistente (default: nuova output_batch_<timestamp>)')
    argparser.add_argument('--senza-camera', action='store_true',
                           help='niente camera ne\' YOLO: prova del moto e dello spectator')
    args = argparser.parse_args()
    camera_enabled = not args.senza_camera

    tests = select_tests(args.tests)

    # Directory dedicate per Scenario 2 (zero interferenza con Scenario 1)
    tl_base_dir = os.path.join(args.batch_dir or new_batch_dir(camera_enabled), 'scenario2_traffic_light')
    tl_results_dir = os.path.join(tl_base_dir, 'results')
    tl_samples_dir = os.path.join(tl_base_dir, 'detection_samples')
    os.makedirs(tl_results_dir, exist_ok=True)
    os.makedirs(tl_samples_dir, exist_ok=True)
    os.makedirs(os.path.join(tl_base_dir, 'plots'), exist_ok=True)

    client, world = connect(args.host, args.port)
    model = load_model()

    print(f"\n{'='*60}")
    print("AVVIO BATCH: SCENARIO 2 (TRAFFIC LIGHT DETECTION)")
    print(f"Cartella output dedicata: {tl_base_dir}")
    if not camera_enabled:
        print("MODALITA' SENZA CAMERA: nessuna detection, solo moto e spectator")
    print(f"{'='*60}")
    print_tests(tests)

    tl_batch_id = f"TL_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tl_summary_rows = []
    tl_frame_rows = []

    for idx, test in enumerate(tests, 1):
        print(f"\n[{idx}/{len(tests)}] TL Test in corso...")

        summary_row, frame_rows = run_condition_tl(
            world=world,
            model=model,
            batch_id=tl_batch_id,
            weather=test['weather'],
            corruption=test['corruption'],
            severity=test['severity'],
            samples_dir=tl_samples_dir,
            camera_enabled=camera_enabled,
        )

        # Red light warning: semaforo confermato e classificato rosso / falso positivo rosso
        summary_row.update(evaluate_warning(frame_rows, 'rlw_positive', 'fp_red', 'rlw_'))
        tl_summary_rows.append(summary_row)
        tl_frame_rows.extend(frame_rows)

        # Salvataggio dopo ogni condizione: un crash non fa perdere i run gia' completati
        pd.DataFrame(tl_summary_rows).to_csv(os.path.join(tl_results_dir, 'run_summary.csv'), index=False)
        pd.DataFrame(tl_frame_rows).to_csv(os.path.join(tl_results_dir, 'frame_metrics.csv'), index=False)

    save_benchmark_config(os.path.join(tl_results_dir, 'config.json'), 'traffic_light', tl_summary_rows, client, world)

    print("\n✅ Scenario 2 completato!")
    print(f"   Risultati salvati in: {tl_results_dir}")
    print(f"   Campioni visivi in:   {tl_samples_dir}")


if __name__ == '__main__':

    main()
