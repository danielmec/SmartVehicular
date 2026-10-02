#!/usr/bin/env python3
"""
CARLA Traffic Light Finder
Trova spawn points con semafori davanti nella direzione di marcia.
Richiede CARLA server attivo su localhost:2000.
"""
import carla
import math

# Connessione
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
carla_map = world.get_map()

print(f"Mappa: {carla_map.name}")
print(f"{'='*70}")

# Prendi tutti i semafori e spawn points
all_tls = list(world.get_actors().filter('traffic.traffic_light'))
spawn_points = carla_map.get_spawn_points()

print(f"Semafori trovati: {len(all_tls)}")
print(f"Spawn points: {len(spawn_points)}")
print(f"{'='*70}\n")

# Per ogni spawn point, cerca semafori DAVANTI (nella direzione di marcia)
candidates = []

for sp_idx, sp in enumerate(spawn_points):
    fwd = sp.get_forward_vector()  # direzione di marcia
    
    for tl in all_tls:
        tl_loc = tl.get_location()
        sp_loc = sp.location
        
        # Vettore dallo spawn al semaforo
        dx = tl_loc.x - sp_loc.x
        dy = tl_loc.y - sp_loc.y
        
        dist = sp_loc.distance(tl_loc)
        
        # Deve essere a distanza utile (40-120m)
        if dist < 40 or dist > 120:
            continue
        
        # Dot product per verificare che sia DAVANTI
        dot = fwd.x * dx + fwd.y * dy
        if dot <= 0:
            continue  # e' dietro
        
        # Angolo laterale (quanto e' centrato rispetto alla direzione di marcia)
        norm_dir = math.sqrt(dx*dx + dy*dy)
        if norm_dir < 0.1:
            continue
        cos_angle = dot / (math.sqrt(fwd.x**2 + fwd.y**2) * norm_dir)
        angle_deg = math.degrees(math.acos(min(1.0, max(-1.0, cos_angle))))
        
        # Il semaforo deve essere abbastanza centrato (< 20 gradi)
        if angle_deg > 20:
            continue
        
        candidates.append({
            'sp_idx': sp_idx,
            'sp_loc': sp_loc,
            'sp_rot': sp.rotation,
            'tl_id': tl.id,
            'tl_loc': tl_loc,
            'dist': dist,
            'angle': angle_deg,
            'tl_state': str(tl.get_state()),
        })

# Ordina per angolo e distanza (preferisci centrati e a 60-80m)
candidates.sort(key=lambda c: (c['angle'], abs(c['dist'] - 80)))

print(f"Candidati trovati: {len(candidates)}\n")

# Mostra i migliori 15
print(f"{'#':>3} {'SP_IDX':>6} {'TL_ID':>6} {'DIST':>7} {'ANGLE':>7} {'TL_STATE':>12}  SPAWN_LOCATION")
print(f"{'-'*80}")

shown = set()
count = 0
for c in candidates:
    key = (c['sp_idx'], c['tl_id'])
    if key in shown:
        continue
    shown.add(key)
    count += 1
    
    sp = c['sp_loc']
    print(f"{count:3d} {c['sp_idx']:6d} {c['tl_id']:6d} {c['dist']:6.1f}m {c['angle']:6.1f}deg {c['tl_state']:>12s}  "
          f"({sp.x:.1f}, {sp.y:.1f}, {sp.z:.1f}) rot=({c['sp_rot'].yaw:.0f}deg)")
    
    if count >= 15:
        break

# Dettaglio del MIGLIORE
if candidates:
    best = candidates[0]
    print(f"\n{'='*70}")
    print(f"MIGLIOR CANDIDATO:")
    print(f"   Spawn Point #{best['sp_idx']}")
    print(f"   Posizione: ({best['sp_loc'].x:.1f}, {best['sp_loc'].y:.1f}, {best['sp_loc'].z:.1f})")
    print(f"   Rotazione: yaw={best['sp_rot'].yaw:.0f}deg")
    print(f"   Semaforo ID: {best['tl_id']}")
    print(f"   Distanza: {best['dist']:.1f}m")
    print(f"   Angolo: {best['angle']:.1f}deg")
    print(f"   Stato: {best['tl_state']}")
    
    # Verifica: lo spawn point 0 (usato per il pedone) e' tra i candidati?
    sp0_candidates = [c for c in candidates if c['sp_idx'] == 0]
    if sp0_candidates:
        print(f"\n   Lo spawn point 0 (usato per il pedone) ha {len(sp0_candidates)} semafori davanti!")
        for c in sp0_candidates[:3]:
            print(f"   TL ID={c['tl_id']}, dist={c['dist']:.1f}m, angle={c['angle']:.1f}deg")
    else:
        print(f"\n   Lo spawn point 0 NON ha semafori davanti (nessun conflitto con Scenario 1)")
else:
    print("\nNessun candidato trovato! Prova con Town03 o Town05 (hanno molti semafori)")
