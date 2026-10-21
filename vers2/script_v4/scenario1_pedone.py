"""
Scenario 1 - Pedone (FCW): esegue il benchmark su tutte le condizioni di TESTS (cella 32 del notebook).

Output in vers2/output_batch_<timestamp>/ (o nella cartella data con --batch-dir):
    results/run_summary.csv, results/frame_metrics.csv, results/config.json
    detection_samples/<run_label>/*.png

I CSV vengono aggiornati dopo ogni condizione; config.json viene scritto solo a batch
completato: l'analisi (analisi_grafici.py) considera validi solo i batch con config.json.

Uso:
    carlapy scenario1_pedone.py                  # tutte le 16 condizioni
    carlapy scenario1_pedone.py --tests 1        # solo clear (prova veloce)
    carlapy scenario1_pedone.py --senza-camera   # solo moto e spectator, senza camera ne' YOLO

Con --senza-camera le metriche di detection sono vuote: l'output va in
vers2/test_senza_camera_<timestamp>/ e non viene letto da analisi_grafici.py.
"""

import argparse
import os
from datetime import datetime

from common import *

SUMMARY_FIELDS = [
    'batch_id', 'run_label', 'weather', 'corruption', 'severity',
    'total_frames', 'detection_frames', 'detection_rate',
    'first_detection_distance_m', 'first_detection_elapsed_s',
    'warning_delay_s', 'min_distance_m',
    'miss_rate', 'gt_visible_frames', 'false_negatives',
    'false_positive_count', 'precision', 'fp_per_frame',
    'critical_zone_frames', 'critical_zone_misses',
    'fcw_warnings_issued', 'fcw_true_positives', 'fcw_missed_warnings',
    'fcw_false_alarms', 'fcw_false_alarm_events', 'fcw_precision', 'fcw_recall',
    'fcw_first_warning_distance_m', 'fcw_ttc_at_warning_s', 'fcw_warning_delay_s',
    'fcw_stopping_distance_m', 'fcw_can_stop', 'fcw_warning_zone_frames',
    'ped_cross_speed_mps', 'other_person_detections', 'ego_collisions',
    'weather_ok', 'weather_params',
    'status',
]

FRAME_FIELDS = [
    'batch_id', 'run_label', 'weather', 'corruption', 'severity',
    'frame_index', 'elapsed_s', 'distance_m',
    'persons_detected', 'max_confidence', 'detected', 'in_critical_zone',
    'gt_visible', 'gt_x1', 'gt_y1', 'gt_x2', 'gt_y2', 'gt_height_px',
    'target_iou', 'tp', 'fn', 'fp', 'fp_confs', 'fp_in_path',
    'speed_mps', 'gap_m', 'ttc_s', 'lateral_offset_m', 'heading_change_deg',
    'ped_lateral_m', 'other_persons_visible', 'other_person_detections',
]


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
    base_output_dir = args.batch_dir or new_batch_dir(camera_enabled)
    results_dir = os.path.join(base_output_dir, 'results')
    samples_dir = os.path.join(base_output_dir, 'detection_samples')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(os.path.join(base_output_dir, 'plots'), exist_ok=True)
    frame_log_path = os.path.join(results_dir, 'frame_metrics.csv')
    run_summary_path = os.path.join(results_dir, 'run_summary.csv')

    client, world = connect(args.host, args.port)
    model = load_model()

    print(f"Configurazione benchmark pronta")
    print(f"   Test da eseguire: {len(tests)}")
    print(f"   Cartella output: {base_output_dir}")
    if not camera_enabled:
        print("   MODALITA' SENZA CAMERA: nessuna detection, solo moto e spectator")
    print()
    print_tests(tests)

    # Genera batch ID univoco
    batch_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_summary_rows = []

    print(f"\nBenchmark avviato | batch={batch_id}")
    print(f"   Test totali: {len(tests)}")
    print("-" * 60)

    for idx, test in enumerate(tests, 1):
        print(f"\n[{idx}/{len(tests)}] Test in corso...")

        summary_row, frame_rows = run_condition(
            world=world,
            model=model,
            batch_id=batch_id,
            weather=test['weather'],
            corruption=test['corruption'],
            severity=test['severity'],
            samples_dir=samples_dir,
            camera_enabled=camera_enabled,
        )

        # Valutazione FCW su frame_rows (pedone confermato / falso positivo nel percorso)
        fcw_summary = evaluate_warning(frame_rows, 'detected', 'fp_in_path', 'fcw_')
        summary_row.update(fcw_summary)
        all_summary_rows.append(summary_row)

        # Salvataggio dopo ogni condizione: un crash non fa perdere i run gia' completati
        append_csv_rows(run_summary_path, SUMMARY_FIELDS, [summary_row])
        append_csv_rows(frame_log_path, FRAME_FIELDS, frame_rows)

    save_benchmark_config(os.path.join(results_dir, 'config.json'), 'pedestrian', all_summary_rows, client, world)

    print("\n" + "=" * 60)
    print(f"Benchmark completato!")
    print(f"   Risultati salvati: {run_summary_path}")
    print(f"   Frame log salvato: {frame_log_path}")
    print(f"   Sample annotate: {samples_dir}")


if __name__ == '__main__':

    main()
