"""
Project 11 - Analisi dei risultati e grafici (celle 36-48 del notebook).

Legge TUTTI i batch validi (con config.json) in vers2/output_batch_*, come nel notebook:
le metriche per condizione sono la media tra i batch. Salva tabelle e grafici nella
cartella plots/ dell'ultimo batch di ogni scenario:
    summary_<ped|tl>.csv/.md, collapse_overview.png, recall_vs_distance.png,
    blur_degradation.png, side_by_side_<ped|tl>.png, failure_cases_<ped|tl>.png

Non serve CARLA ne' il server acceso.

Uso:
    carlapy analisi_grafici.py
"""

import glob
import json
import os
import re

import matplotlib
matplotlib.use('Agg')   # niente finestre: i grafici vengono solo salvati su file
import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

# Stile plot (cella 5)
plt.style.use('seaborn-whitegrid')
sns.set_palette("husl")

# I percorsi output_batch_* sono relativi alla cartella vers2, come nel notebook
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def gt_zoom(image, frame_row, scale=3):
    """Ritaglio ingrandito attorno al box di ground truth di un frame (cella 19, copia di common.gt_zoom)."""
    x1, y1, x2, y2 = (frame_row[k] for k in ('gt_x1', 'gt_y1', 'gt_x2', 'gt_y2'))
    margin = max(40.0, 1.5 * (y2 - y1))
    h, w = image.shape[:2]
    cx1, cy1 = int(max(0, x1 - margin)), int(max(0, y1 - margin))
    cx2, cy2 = int(min(w, x2 + margin)), int(min(h, y2 + margin))
    crop = image[cy1:cy2, cx1:cx2]
    return cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale), interpolation=cv2.INTER_NEAREST)


# ============================================================================
# -- Cella 36 del notebook ------------------------------------------------------
# ============================================================================

# CARICAMENTO RISULTATI E TABELLA RIASSUNTIVA
COLLAPSE_RECALL_FRAC = 0.5   # COLLASSO se la recall in zona critica < 50% di quella in clear
DEGRADED_RECALL_FRAC = 0.8   # DEGRADATO se < 80%
DEGRADED_TTC_DROP_S = 0.5    # DEGRADATO se il warning arriva > 0.5 s piu' tardi (TTC) rispetto al clear

SCENARIOS = {
    'ped': {'name': 'Scenario 1 - Pedone (FCW)', 'results': 'results', 'samples': 'detection_samples',
            'prefix': 'fcw_', 'side_sample': '04_critical'},
    'tl': {'name': 'Scenario 2 - Semaforo (Red Light Warning)',
           'results': os.path.join('scenario2_traffic_light', 'results'),
           'samples': os.path.join('scenario2_traffic_light', 'detection_samples'),
           'prefix': 'rlw_', 'side_sample': '03_approach'},
}
SEVERITY_NAMES = {1: 'σ1', 2: 'σ2', 3: 'mild', 4: 'severe'}   # blur: sigma 1, 2, 3, 7 px a 1280 px
VERDICT_COLORS = {'OK': '#2e7d32', 'DEGRADATO': '#f9a825', 'COLLASSO': '#c62828'}


def condition_label(weather, corruption, severity):
    """Nome leggibile della condizione: 'fog', 'blur mild', 'rain + blur severe', ..."""
    weather = weather.replace('_', ' ')   # 'fog_severe' -> 'fog severe'
    if corruption == 'none':
        return weather
    name = f"{corruption.replace('_', ' ')} {SEVERITY_NAMES.get(int(severity), f's{int(severity)}')}"
    return name if weather == 'clear' else f"{weather} + {name}"


def load_scenario(key):
    """Carica summary e frame di tutti i batch validi (con config.json) di uno scenario."""
    cfg = SCENARIOS[key]
    batches = sorted(d for d in glob.glob('output_batch_*')
                     if os.path.exists(os.path.join(d, cfg['results'], 'config.json')))
    if not batches:
        return None
    summaries, frames = [], []
    for d in batches:
        s = pd.read_csv(os.path.join(d, cfg['results'], 'run_summary.csv'))
        s['batch_dir'] = d
        summaries.append(s)
        f_path = os.path.join(d, cfg['results'], 'frame_metrics.csv')
        if os.path.exists(f_path):
            f = pd.read_csv(f_path)
            f['batch_dir'] = d
            frames.append(f)
    summary = pd.concat(summaries, ignore_index=True)
    frames = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    for df in (summary, frames):
        if len(df):
            df['condition'] = [condition_label(w, c, s) for w, c, s in zip(df['weather'], df['corruption'], df['severity'])]
    with open(os.path.join(batches[-1], cfg['results'], 'config.json'), encoding='utf-8') as fh:
        config = json.load(fh)
    return {'summary': summary, 'frames': frames, 'batches': batches, 'latest': batches[-1], 'config': config}


def aggregate(summary, prefix):
    """Una riga per condizione (media tra i batch), nell'ordine dei TESTS."""
    ok = summary[summary['status'] == 'ok'].copy()
    ok['critical_recall'] = np.where(ok['critical_zone_frames'] > 0, 1 - ok['miss_rate'], np.nan)
    ok['can_stop_num'] = ok[prefix + 'can_stop'].map({True: 1.0, False: 0.0, 'True': 1.0, 'False': 0.0})
    for col in (prefix + 'ttc_at_warning_s', prefix + 'warning_delay_s', 'first_detection_distance_m'):
        ok[col] = pd.to_numeric(ok[col], errors='coerce')
    named = {
        'n_run': ('batch_dir', 'nunique'),
        'critical_recall': ('critical_recall', 'mean'),
        'critical_recall_std': ('critical_recall', 'std'),
        'recall': ('detection_rate', 'mean'),
        'first_detection_m': ('first_detection_distance_m', 'mean'),
        'precision': ('precision', 'mean'),
        'fp_per_frame': ('fp_per_frame', 'mean'),
        'ttc_at_warning_s': (prefix + 'ttc_at_warning_s', 'mean'),
        'warning_delay_s': (prefix + 'warning_delay_s', 'mean'),
        'can_stop': ('can_stop_num', 'mean'),
    }
    if 'state_accuracy' in ok:
        named['state_accuracy'] = ('state_accuracy', 'mean')
    order = list(dict.fromkeys(ok['condition']))
    return ok.groupby('condition', sort=False).agg(**named).reindex(order)


def verdict(row, ref):
    """Esito (OK / DEGRADATO / COLLASSO) e motivo, rispetto alla riga di riferimento (clear)."""
    if not row['can_stop'] >= 0.5:
        return 'COLLASSO', ('warning troppo tardi per fermarsi' if pd.notna(row['ttc_at_warning_s'])
                            else 'nessun warning')
    if ref is None:
        return 'OK', ''
    if ref['critical_recall'] > 0:
        ratio = row['critical_recall'] / ref['critical_recall']
        if ratio < COLLAPSE_RECALL_FRAC:
            return 'COLLASSO', f'recall zona critica al {ratio:.0%} del clear'
        if ratio < DEGRADED_RECALL_FRAC:
            return 'DEGRADATO', f'recall zona critica al {ratio:.0%} del clear'
    ttc_drop = ref['ttc_at_warning_s'] - row['ttc_at_warning_s']
    if ttc_drop > DEGRADED_TTC_DROP_S:
        return 'DEGRADATO', f'warning {ttc_drop:.1f} s piu\' tardi del clear'
    return 'OK', ''


RESULTS = {}
for key, cfg in SCENARIOS.items():
    res = load_scenario(key)
    if res is None:
        print(f"{cfg['name']}: nessun batch valido trovato (eseguire le Sezioni 10/11)")
        continue
    agg = aggregate(res['summary'], cfg['prefix'])
    ref = agg.loc['clear'] if 'clear' in agg.index else None
    verdicts = [verdict(row, ref) for _, row in agg.iterrows()]
    agg['esito'] = [v[0] for v in verdicts]
    agg['motivo'] = [v[1] for v in verdicts]
    res['agg'] = agg
    res['plots_dir'] = os.path.join(res['latest'], 'plots')
    os.makedirs(res['plots_dir'], exist_ok=True)
    RESULTS[key] = res

    pct = lambda v: f'{v:.0%}' if pd.notna(v) else '-'
    table = pd.DataFrame({
        'Condizione': agg.index,
        'Recall zona critica': agg['critical_recall'].map(pct),
        'Recall totale': agg['recall'].map(pct),
        'Prima detection (m)': agg['first_detection_m'].map(lambda v: f'{v:.1f}' if pd.notna(v) else '-'),
        'Precision': agg['precision'].map(pct),
        'FP/frame': agg['fp_per_frame'].map('{:.3f}'.format),
        'TTC al warning (s)': agg['ttc_at_warning_s'].map(lambda v: f'{v:.2f}' if pd.notna(v) else 'nessuno'),
        'Ritardo warning (s)': agg['warning_delay_s'].map(lambda v: f'{v:.2f}' if pd.notna(v) else '-'),
        'Si ferma in tempo': agg['can_stop'].map(lambda v: 'si' if v >= 0.5 else 'NO'),
        'Esito': agg['esito'],
        'Motivo': agg['motivo'],
    })
    if 'state_accuracy' in agg:
        table.insert(5, 'Colore corretto', agg['state_accuracy'].map(pct).values)
    res['table'] = table

    print('=' * 100)
    print(f"  {cfg['name']}  |  batch: {', '.join(res['batches'])}  |  run per condizione: {int(agg['n_run'].max())}")
    print('=' * 100)
    print(table.to_string(index=False))
    print()
    table.to_csv(os.path.join(res['plots_dir'], f'summary_{key}.csv'), index=False)
    try:
        with open(os.path.join(res['plots_dir'], f'summary_{key}.md'), 'w', encoding='utf-8') as fh:
            fh.write(table.to_markdown(index=False))
    except ImportError:
        pass  # to_markdown richiede il pacchetto 'tabulate'


# ============================================================================
# -- Cella 38 del notebook ------------------------------------------------------
# ============================================================================

# QUANDO COLLASSA: recall in zona critica e TTC al warning per condizione
def warning_limits(config):
    """(soglia TTC di warning, TTC minimo per fermarsi) dalla configurazione del batch."""
    w, m = config['warning'], config['motion']
    speed = m['ego_speed_kmh'] / 3.6
    return w['ttc_warning_s'], w['reaction_s'] + speed / (2.0 * w['decel_mps2'])


def verdict_legend(ax, extra=()):
    handles = [mpatches.Patch(color=c, label=v) for v, c in VERDICT_COLORS.items()]
    ax.legend(handles=handles + list(extra), fontsize=9, loc='upper right', ncol=2)


if RESULTS:
    fig, axes = plt.subplots(len(RESULTS), 2, figsize=(16, 5 * len(RESULTS)), squeeze=False)
    for (key, res), (ax_rec, ax_ttc) in zip(RESULTS.items(), axes):
        agg, cfg = res['agg'], SCENARIOS[key]
        x = np.arange(len(agg))
        colors = [VERDICT_COLORS[v] for v in agg['esito']]
        ttc_warn, ttc_stop = warning_limits(res['config'])

        # Recall in zona critica
        yerr = agg['critical_recall_std'] if agg['n_run'].max() > 1 else None
        ax_rec.bar(x, agg['critical_recall'], color=colors, yerr=yerr, capsize=4)
        extra = []
        if 'clear' in agg.index:
            thr = COLLAPSE_RECALL_FRAC * agg.loc['clear', 'critical_recall']
            extra.append(ax_rec.axhline(thr, color='#c62828', ls='--', lw=1.5,
                                        label=f'soglia collasso ({COLLAPSE_RECALL_FRAC:.0%} del clear)'))
        ax_rec.set_ylim(0, 1.35)  # spazio in alto per la legenda
        ax_rec.set_ylabel('Recall in zona critica')
        ax_rec.set_title(f"{cfg['name']}\nrecall in zona critica", fontweight='bold')
        verdict_legend(ax_rec, extra)

        # TTC al primo warning (0 = nessun warning)
        ttc = agg['ttc_at_warning_s']
        ax_ttc.bar(x, ttc.fillna(0), color=colors)
        for xi, v in zip(x, ttc):
            if pd.isna(v):
                ax_ttc.text(xi, 0.05, 'nessun\nwarning', ha='center', va='bottom', fontsize=8, color='#c62828')
        lines = [ax_ttc.axhline(ttc_stop, color='#c62828', ls='--', lw=1.5,
                                label=f'TTC minimo per fermarsi ({ttc_stop:.2f} s)'),
                 ax_ttc.axhline(ttc_warn, color='gray', ls=':', lw=1.5,
                                label=f'soglia di warning ({ttc_warn:.1f} s)')]
        ax_ttc.set_ylim(0, ttc_warn * 1.5)  # spazio in alto per la legenda
        ax_ttc.set_ylabel('TTC al primo warning (s)')
        ax_ttc.set_title(f"{cfg['name']}\nTTC al primo warning", fontweight='bold')
        verdict_legend(ax_ttc, lines)

        for ax in (ax_rec, ax_ttc):
            ax.set_xticks(x)
            ax.set_xticklabels(agg.index, rotation=30, ha='right')
            ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    for plots_dir in {r['plots_dir'] for r in RESULTS.values()}:
        plt.savefig(os.path.join(plots_dir, 'collapse_overview.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# -- Cella 40 del notebook ------------------------------------------------------
# ============================================================================

# RECALL IN FUNZIONE DELLA DISTANZA (bin da 5 m)
if RESULTS:
    fig, axes = plt.subplots(1, len(RESULTS), figsize=(9 * len(RESULTS), 6), squeeze=False)
    for (key, res), ax in zip(RESULTS.items(), axes[0]):
        frames = res['frames']
        if frames.empty:
            continue
        vis = frames[frames['gt_visible'] == 1].copy()
        max_d = float(np.ceil(vis['distance_m'].max() / 5.0) * 5.0)
        crit_m = vis.loc[vis['in_critical_zone'] == 1, 'distance_m'].max()  # limite della zona critica
        vis['bin'] = pd.cut(vis['distance_m'], bins=np.arange(0, max_d + 5, 5), right=False)
        cmap = plt.get_cmap('tab20')
        for i, cond in enumerate(res['agg'].index):
            curve = vis[vis['condition'] == cond].groupby('bin', observed=True)['tp'].mean()
            ax.plot([b.mid for b in curve.index], curve.values, marker='o', ms=4, lw=2,
                    color=cmap(i % 20), label=cond)
        if not pd.isna(crit_m):
            ax.axvspan(0, crit_m, color='#c62828', alpha=0.07, label='zona critica')
        ax.set_xlim(max_d, 0)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("Distanza dalla linea d'arresto (m)" if key == 'tl' else 'Distanza dal pedone (m)')
        ax.set_ylabel('Recall')
        ax.set_title(f"{SCENARIOS[key]['name']}\nrecall per distanza", fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='upper left', ncol=2)
    plt.tight_layout()
    for plots_dir in {r['plots_dir'] for r in RESULTS.values()}:
        plt.savefig(os.path.join(plots_dir, 'recall_vs_distance.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# -- Cella 42 del notebook ------------------------------------------------------
# ============================================================================

# BLUR: clear -> sigma 1 -> sigma 2 -> mild (sigma 3) -> severe (sigma 7)
BLUR_STEPS = ['clear', 'blur σ1', 'blur σ2', 'blur mild', 'blur severe']
BLUR_TICKS = ['nessuna (clear)', 'σ = 1 px', 'σ = 2 px', 'mild (σ = 3 px)', 'severe (σ = 7 px)']

blur_results = {k: r for k, r in RESULTS.items() if all(s in r['agg'].index for s in BLUR_STEPS)}
if blur_results:
    fig, (ax_rec, ax_ttc) = plt.subplots(1, 2, figsize=(16, 5.5))
    markers = {'ped': 'o', 'tl': 's'}
    x = np.arange(len(BLUR_STEPS))
    for key, res in blur_results.items():
        rows = res['agg'].loc[BLUR_STEPS]
        name = SCENARIOS[key]['name']
        ax_rec.plot(x, rows['critical_recall'], marker=markers[key], ms=9, lw=2.5, label=name)
        ax_rec.axhline(COLLAPSE_RECALL_FRAC * rows.loc['clear', 'critical_recall'], ls='--', lw=1, alpha=0.6,
                       color=ax_rec.lines[-1].get_color(), label=f'soglia collasso ({name.split(" - ")[0]})')
        ax_ttc.plot(x, rows['ttc_at_warning_s'], marker=markers[key], ms=9, lw=2.5, label=name)
        for xi, (v, esito) in enumerate(zip(rows['ttc_at_warning_s'], rows['esito'])):
            ax_ttc.annotate(esito, (xi, 0 if pd.isna(v) else v), textcoords='offset points', xytext=(0, 8),
                            ha='center', fontsize=8, color=VERDICT_COLORS[esito])
    ttc_warn, ttc_stop = warning_limits(next(iter(blur_results.values()))['config'])
    ax_ttc.axhline(ttc_stop, color='#c62828', ls='--', lw=1.5, label=f'TTC minimo per fermarsi ({ttc_stop:.2f} s)')
    ax_rec.set_ylabel('Recall in zona critica')
    ax_rec.set_ylim(0, 1.05)
    ax_ttc.set_ylabel('TTC al primo warning (s)')
    ax_ttc.set_ylim(0, ttc_warn * 1.25)
    for ax, title in ((ax_rec, 'Recall in zona critica'), (ax_ttc, 'TTC al primo warning')):
        ax.set_xticks(x)
        ax.set_xticklabels(BLUR_TICKS)
        ax.set_title(f'Blur: {title}', fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    plt.tight_layout()
    for plots_dir in {r['plots_dir'] for r in blur_results.values()}:
        plt.savefig(os.path.join(plots_dir, 'blur_degradation.png'), dpi=150, bbox_inches='tight')
    plt.close()
else:
    print(f"Condizioni {' / '.join(BLUR_STEPS)} non presenti nei risultati")


# ============================================================================
# -- Cella 44 del notebook ------------------------------------------------------
# ============================================================================

# SIDE-BY-SIDE: stesso frame (stessa distanza) in tutte le condizioni, dall'ultimo batch
import re


def sample_with_gt(res, run_label, prefix):
    """(immagine annotata, ingrandimento sul GT, distanza) del sample che inizia con prefix, o None."""
    run_dir = os.path.join(res['latest'], SCENARIOS[res['key']]['samples'], run_label)
    if not os.path.isdir(run_dir):
        return None
    for fname in sorted(os.listdir(run_dir)):
        m = re.match(re.escape(prefix) + r'.*?_(\d+\.\d)m', fname)
        if not m:
            continue
        image = np.array(Image.open(os.path.join(run_dir, fname)).convert('RGB'))
        frames = res['frames']
        vis = frames[(frames['batch_dir'] == res['latest']) & (frames['run_label'] == run_label)
                     & (frames['gt_visible'] == 1)]
        zoom = None
        if len(vis):
            row = vis.iloc[(vis['distance_m'] - float(m.group(1))).abs().argmin()]
            zoom = gt_zoom(image, row)
        return image, zoom, float(m.group(1))
    return None


def panel_axes(n, col_w, row_h, max_cols=6):
    """Figura con n pannelli (immagine sopra, ingrandimento sotto), al massimo max_cols per riga."""
    ncols = min(n, max_cols)
    blocks = -(-n // ncols)
    fig, axes = plt.subplots(2 * blocks, ncols, figsize=(col_w * ncols, row_h * blocks), squeeze=False,
                             gridspec_kw={'height_ratios': [1, 1.3] * blocks})
    for ax in axes.flat:
        ax.axis('off')
    return fig, [(axes[2 * (i // ncols), i % ncols], axes[2 * (i // ncols) + 1, i % ncols]) for i in range(n)]


for key, res in RESULTS.items():
    res['key'] = key
    latest = res['summary'][res['summary']['batch_dir'] == res['latest']]
    labels = dict(zip(latest['condition'], latest['run_label']))
    panels = []
    for cond in res['agg'].index:
        if cond in labels:
            found = sample_with_gt(res, labels[cond], SCENARIOS[key]['side_sample'])
            if found:
                panels.append((cond, res['agg'].loc[cond, 'esito']) + found)
    if not panels:
        print(f"{SCENARIOS[key]['name']}: nessun sample trovato in {res['latest']}")
        continue
    fig, pairs = panel_axes(len(panels), 3.6, 5.2)
    for (ax_img, ax_zoom), (cond, esito, image, zoom, dist) in zip(pairs, panels):
        ax_img.imshow(image)
        ax_img.set_title(f"{cond}\n{esito} - {dist:.0f} m", fontsize=10, fontweight='bold',
                         color=VERDICT_COLORS[esito])
        if zoom is not None:
            ax_zoom.imshow(zoom)
    plt.suptitle(f"{SCENARIOS[key]['name']}: stesso frame in tutte le condizioni", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(res['plots_dir'], f'side_by_side_{key}.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# -- Cella 46 del notebook ------------------------------------------------------
# ============================================================================

# FAILURE CASES: frame rilevati in clear ma persi nella condizione (ultimo batch)
for key, res in RESULTS.items():
    frames = res['frames']
    latest = frames[frames['batch_dir'] == res['latest']]
    clear = latest[latest['condition'] == 'clear'].set_index('frame_index')
    if clear.empty:
        print(f"{SCENARIOS[key]['name']}: condizione clear assente, confronto non possibile")
        continue
    rows = []
    for cond in res['agg'].index:
        if cond == 'clear':
            continue
        other = latest[latest['condition'] == cond].set_index('frame_index')
        both = clear[['tp', 'distance_m']].join(other[['tp', 'gt_visible', 'in_critical_zone']],
                                                 rsuffix='_cond', how='inner')
        lost = both[(both['tp'] == 1) & (both['tp_cond'] == 0) & (both['gt_visible'] == 1)]
        lost_crit = lost[lost['in_critical_zone'] == 1]
        rows.append({
            'Condizione': cond,
            'Esito': res['agg'].loc[cond, 'esito'],
            'Frame persi rispetto al clear': f"{len(lost)} / {int(both['tp'].sum())}",
            'Distanze dei frame persi (m)': (f"{lost['distance_m'].min():.1f} - {lost['distance_m'].max():.1f}"
                                            if len(lost) else '-'),
            'Persi in zona critica': len(lost_crit),
            'Primo perso in zona critica (m)': f"{lost_crit['distance_m'].max():.1f}" if len(lost_crit) else '-',
        })
    print('=' * 100)
    print(f"  FAILURE CASES - {SCENARIOS[key]['name']}")
    print('=' * 100)
    print(pd.DataFrame(rows).to_string(index=False))
    print()

    # Immagini: primo miss in zona critica per le condizioni non OK
    res['key'] = key
    summary_latest = res['summary'][res['summary']['batch_dir'] == res['latest']]
    labels = dict(zip(summary_latest['condition'], summary_latest['run_label']))
    failing = [c for c in res['agg'].index if res['agg'].loc[c, 'esito'] != 'OK' and c in labels]
    panels = []
    for cond in failing:
        found = sample_with_gt(res, labels[cond], 'miss_crit')
        if found:
            panels.append((cond, found))
    if panels:
        fig, pairs = panel_axes(len(panels), 4.2, 5.6)
        for (ax_img, ax_zoom), (cond, (image, zoom, dist)) in zip(pairs, panels):
            esito = res['agg'].loc[cond, 'esito']
            ax_img.imshow(image)
            ax_img.set_title(f"{cond} ({esito})\nprimo miss in zona critica: {dist:.1f} m", fontsize=10,
                             fontweight='bold', color=VERDICT_COLORS[esito])
            if zoom is not None:
                ax_zoom.imshow(zoom)
        plt.suptitle(f"Failure cases - {SCENARIOS[key]['name']}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(res['plots_dir'], f'failure_cases_{key}.png'), dpi=150, bbox_inches='tight')
        plt.close()


# ============================================================================
# -- Cella 48 del notebook ------------------------------------------------------
# ============================================================================

# SINTESI AUTOMATICA
for key, res in RESULTS.items():
    agg = res['agg']
    print('=' * 80)
    print(f"  {SCENARIOS[key]['name']}")
    print('=' * 80)
    for esito in ('COLLASSO', 'DEGRADATO', 'OK'):
        rows = agg[agg['esito'] == esito]
        if len(rows):
            items = [f"{c} ({m})" if m else c for c, m in zip(rows.index, rows['motivo'])]
            print(f"  {esito:10s}: {', '.join(items)}")

    others = agg.drop(index='clear', errors='ignore')
    if len(others) and others['critical_recall'].notna().any():
        worst = others['critical_recall'].idxmin()
        print(f"  Condizione peggiore (recall in zona critica): {worst} "
              f"= {agg.loc[worst, 'critical_recall']:.0%}"
              + (f" contro {agg.loc['clear', 'critical_recall']:.0%} in clear" if 'clear' in agg.index else ''))

    steps = [s for s in BLUR_STEPS if s in agg.index]
    if len(steps) == len(BLUR_STEPS):
        esiti = agg.loc[steps, 'esito'].tolist()
        first_bad = next((s for s, e in zip(steps, esiti) if e == 'COLLASSO'), None)
        if first_bad:
            print(f"  Blur: il sistema collassa a partire da '{first_bad}' "
                  f"(esiti: {', '.join(f'{s}={e}' for s, e in zip(steps, esiti))})")
        else:
            print(f"  Blur: nessun collasso fino a blur severe (esiti: {', '.join(esiti)})")
    print()

