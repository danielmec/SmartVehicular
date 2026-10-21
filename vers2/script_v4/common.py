"""
Project 11: Adverse Weather Robustness Benchmark (v4.0) - codice comune.

Configurazione, meteo, scenari, utility, ground truth, modulo di warning e le funzioni
che eseguono una singola condizione (run_condition per lo Scenario 1, run_condition_tl
per lo Scenario 2). Il codice e' copiato dal notebook Project11_Adverse_Weather_Benchmark_v4.ipynb
(il numero della cella di origine e' indicato in ogni sezione).

Usato da scenario1_pedone.py e scenario2_semaforo.py; i grafici sono in analisi_grafici.py.
"""

# FIX CRUCIALE PER WINDOWS: OpenMP e PyTorch PRIMA di matplotlib/seaborn
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

import csv
import json
import queue
import sys
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from ultralytics import YOLO

import carla


# Cartella vers2: modello e output_batch_* stanno qui come nel notebook,
# indipendentemente dalla cartella da cui si lanciano gli script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Connessione (cella 7)
CARLA_HOST = 'localhost'
CARLA_PORT = 2000
CARLA_TIMEOUT = 20.0

# Modello (cella 13): percorso relativo a BASE_DIR, salvato cosi' anche in config.json
MODEL_PATH = 'yolov8n.pt'

# Mappa su cui sono calibrati gli spawn point (0 = pedone, 103 = semaforo)
EXPECTED_MAP = 'Town10HD_Opt'


def connect(host=CARLA_HOST, port=CARLA_PORT):
    """Connessione al server CARLA (cella 7). Esce dallo script se il server non risponde."""
    try:
        client = carla.Client(host, port)
        client.set_timeout(CARLA_TIMEOUT)
        world = client.get_world()
    except RuntimeError as e:
        print(f"Errore di connessione: {e}")
        print(f"   Verifica che CARLA sia in esecuzione su {host}:{port}")
        sys.exit(1)

    print(f" Connesso a CARLA")
    print(f"   Client version: {client.get_client_version()}")
    print(f"   Server version: {client.get_server_version()}")
    print(f"   Mappa corrente: {world.get_map().name}")
    if not world.get_map().name.endswith(EXPECTED_MAP):
        print(f"   [WARN] gli spawn point sono calibrati su {EXPECTED_MAP}: caricare quella mappa")
    print(f"   Accelerazione YOLO: {select_device()}")
    return client, world


def load_model():
    """Carica YOLOv8n (cella 13)."""
    # Le versioni recenti di ultralytics stampano "'half' is deprecated" a ogni frame (FP16 funziona lo stesso):
    # si nasconde solo quel messaggio
    try:
        from ultralytics.utils import LOGGER as _YOLO_LOGGER
        _YOLO_LOGGER.addFilter(lambda record: "'half' is deprecated" not in record.getMessage())
    except Exception:
        pass

    model = YOLO(os.path.join(BASE_DIR, MODEL_PATH))
    print(f"Modello YOLOv8n caricato da {MODEL_PATH}")
    return model


def new_batch_dir(camera_enabled=True):
    """
    Nuova cartella output_batch_<timestamp> in BASE_DIR (cella 21). Senza camera la cartella
    si chiama test_senza_camera_<timestamp>: analisi_grafici.py legge solo output_batch_*.
    """
    batch_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix = 'output_batch' if camera_enabled else 'test_senza_camera'
    return os.path.join(BASE_DIR, f'{prefix}_{batch_timestamp}')


def print_tests(tests):
    """Elenco dei test configurati (cella 21)."""
    print("Test configurati:")
    for i, t in enumerate(tests, 1):
        print(f"   [{i:2d}] weather={t['weather']:10s} corruption={t['corruption']:12s} severity={t['severity']}")


def select_tests(spec):
    """
    Sottoinsieme dei TESTS da eseguire: None = tutti, oppure numeri 1-based separati da
    virgola come nell'elenco di print_tests (es. '1,9').
    """
    if not spec:
        return list(TESTS)
    indices = [int(s) for s in spec.split(',')]
    for i in indices:
        if not 1 <= i <= len(TESTS):
            raise SystemExit(f"Test {i} inesistente: i test vanno da 1 a {len(TESTS)}")
    return [TESTS[i - 1] for i in indices]


# ============================================================================
# -- Cella 9 del notebook -------------------------------------------------------
# ============================================================================

# Ordine e nomi dei parametri di carla.WeatherParameters (CARLA 0.9.16)
WEATHER_FIELDS = (
    'cloudiness', 'precipitation', 'precipitation_deposits', 'wind_intensity',
    'sun_azimuth_angle', 'sun_altitude_angle',
    'fog_density', 'fog_distance', 'fog_falloff', 'wetness',
    'scattering_intensity', 'mie_scattering_scale', 'rayleigh_scattering_scale',
    'dust_storm',
)

# Base comune (= clear): ogni condizione sovrascrive solo cio' che la caratterizza
BASE_WEATHER = {
    'cloudiness': 0.0,
    'precipitation': 0.0,
    'precipitation_deposits': 0.0,
    'wind_intensity': 0.0,
    'sun_azimuth_angle': 0.0,
    'sun_altitude_angle': 70.0,
    'fog_density': 0.0,
    'fog_distance': 0.0,
    'fog_falloff': 0.0,
    'wetness': 0.0,
    'scattering_intensity': 1.0,
    'mie_scattering_scale': 0.03,
    'rayleigh_scattering_scale': 0.0331,
    'dust_storm': 0.0,
}

WEATHER_OVERRIDES = {
    'clear': {},
    'rain': {
        'cloudiness': 90.0,
        'precipitation': 95.0,
        'precipitation_deposits': 90.0,
        'wind_intensity': 100.0,
        'wetness': 100.0,
        'fog_density': 8.0,
    },
    'fog': {
        'fog_density': 45.0,
        'fog_distance': 25.0,
        'fog_falloff': 1.5,
    },
    'night': {
        'sun_altitude_angle': -25.0,
    },
    # --- Livelli severi: stessa condizione, molto piu' pesante ---
    'rain_severe': {               # temporale al crepuscolo: pioggia massima, cielo coperto, poca luce
        'cloudiness': 100.0,
        'precipitation': 100.0,
        'precipitation_deposits': 100.0,
        'wind_intensity': 100.0,
        'wetness': 100.0,
        'fog_density': 25.0,
        'fog_distance': 10.0,
        'sun_altitude_angle': 3.0,
    },
    'fog_60': {                    # nebbia intermedia tra fog e fog_severe (soglia di fallimento)
        'fog_density': 60.0,
        'fog_distance': 15.0,
        'fog_falloff': 1.5,
    },
    'fog_75': {
        'fog_density': 75.0,
        'fog_distance': 10.0,
        'fog_falloff': 1.5,
    },
    'fog_severe': {                # nebbia fittissima: visibilita' di pochi metri
        'fog_density': 90.0,
        'fog_distance': 5.0,
        'fog_falloff': 1.5,
    },
    'night_severe': {              # notte senza lampioni (vedi STREET_LIGHTS_OFF): solo i fari dell'ego
        'sun_altitude_angle': -25.0,
        'cloudiness': 80.0,
    },
    # --- Combinazioni ---
    'night_highbeam': {            # come night_severe, ma con gli abbaglianti dell'ego (vedi EGO_HIGH_BEAM)
        'sun_altitude_angle': -25.0,
        'cloudiness': 80.0,
    },
    'rain_night': {                # pioggia di 'rain' di notte, con i lampioni accesi
        'cloudiness': 90.0,
        'precipitation': 95.0,
        'precipitation_deposits': 90.0,
        'wind_intensity': 100.0,
        'wetness': 100.0,
        'fog_density': 8.0,
        'sun_altitude_angle': -25.0,
    },
    'backlight': {
        'sun_azimuth_angle': 180.0,  # sostituito da (yaw ego + offset dello spawn) quando lo yaw e' noto
        'sun_altitude_angle': 8.0,   # sole basso: entro il campo visivo verticale della camera
    },
}

# Condizioni in cui i lampioni della citta' restano SPENTI anche di notte
STREET_LIGHTS_OFF = {'night_severe', 'night_highbeam'}

# Condizioni in cui l'ego accende gli abbaglianti (nelle altre: luci di posizione + anabbaglianti)
EGO_HIGH_BEAM = {'night_highbeam'}

# Azimut del sole relativo alla direzione di marcia dell'ego per avere il sole DAVANTI
# alla camera (controluce), per spawn point: l'offset in cui il sole compare
# nell'inquadratura. Entrambi gli spawn hanno yaw ~0: con 180 il sole e' davanti
# (con 0 il sole era alle spalle dell'ego nello Scenario 2).
BACKLIGHT_AZIMUTH_OFFSET = {0: 180.0, 103: 180.0}   # {spawn point: offset in gradi}


def backlight_offset(spawn_index):
    """Offset di azimut del sole per lo spawn point (180 se lo spawn non e' calibrato)."""
    return BACKLIGHT_AZIMUTH_OFFSET.get(spawn_index, 180.0)


def weather_params(weather_type, ego_yaw=None, spawn_index=None):
    """
    Dizionario completo dei parametri meteo per una condizione.

    Args:
        weather_type (str): una chiave di WEATHER_OVERRIDES ('clear', 'fog', 'fog_severe', ...)
        ego_yaw (float): yaw (gradi) dell'ego vehicle; per 'backlight' orienta il sole
                         davanti alla camera
        spawn_index (int): spawn point dell'ego, per l'offset calibrato del backlight
    """
    weather_type = weather_type.lower()
    if weather_type not in WEATHER_OVERRIDES:
        raise ValueError(f"Condizione meteo sconosciuta: {weather_type}")
    params = dict(BASE_WEATHER)
    params.update(WEATHER_OVERRIDES[weather_type])
    if weather_type == 'backlight' and ego_yaw is not None:
        params['sun_azimuth_angle'] = round((ego_yaw + backlight_offset(spawn_index)) % 360.0, 2)
    return params


def set_street_lights(world, weather_type, params):
    """
    Lampioni della citta' in modo esplicito e ripetibile: accesi solo di notte (sole sotto
    l'orizzonte) e spenti nelle condizioni di STREET_LIGHTS_OFF. Il ciclo giorno/notte
    automatico di CARLA viene disattivato, altrimenti riaccenderebbe le luci da solo.
    Da chiamare DOPO world.set_weather(). Restituisce 'on', 'off' o 'error'.
    """
    on = params['sun_altitude_angle'] < 0 and weather_type not in STREET_LIGHTS_OFF
    try:
        lm = world.get_lightmanager()
        lm.set_day_night_cycle(False)
        lights = lm.get_all_lights(carla.LightGroup.Street)
        if on:
            lm.turn_on(lights)
        else:
            lm.turn_off(lights)
        return 'on' if on else 'off'
    except Exception as exc:
        print(f"  [WARN] impossibile impostare i lampioni: {exc}")
        return 'error'


def set_ego_lights(vehicle, weather_type):
    """
    Luci dell'ego: posizione + anabbaglianti, piu' gli abbaglianti nelle condizioni di EGO_HIGH_BEAM.
    Restituisce 'high_beam' o 'low_beam' (salvato nei parametri del run).
    """
    state = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam
    high = weather_type in EGO_HIGH_BEAM
    if high:
        state |= carla.VehicleLightState.HighBeam
    vehicle.set_light_state(carla.VehicleLightState(state))
    return 'high_beam' if high else 'low_beam'


def build_weather(weather_type, ego_yaw=None, spawn_index=None):
    """
    Costruisce parametri meteo CARLA per una condizione specifica.
    Restituisce sempre un oggetto NUOVO (mai un preset condiviso di CARLA).

    Args:
        weather_type (str): una chiave di WEATHER_OVERRIDES
        ego_yaw, spawn_index: usati solo da 'backlight' (vedi weather_params)

    Returns:
        carla.WeatherParameters
    """
    return carla.WeatherParameters(**weather_params(weather_type, ego_yaw, spawn_index))


def weather_to_dict(weather):
    """Legge tutti i parametri da un carla.WeatherParameters (es. world.get_weather())."""
    return {f: round(float(getattr(weather, f)), 4) for f in WEATHER_FIELDS}


def check_applied_weather(world, expected, label='', tol=1e-3):
    """
    Confronta il meteo attivo sul server con i parametri attesi (dict di weather_params).

    Returns:
        bool: True se tutti i parametri coincidono (angoli confrontati modulo 360)
    """
    applied = weather_to_dict(world.get_weather())
    diff = {}
    for k in WEATHER_FIELDS:   # le chiavi extra ('street_lights', 'ego_lights') non sono parametri meteo
        delta = abs(expected[k] - applied[k])
        if k == 'sun_azimuth_angle':
            delta = min(delta % 360.0, 360.0 - delta % 360.0)
        if delta > tol:
            diff[k] = (expected[k], applied[k])
    if diff:
        print(f"  [WARN] Meteo applicato diverso da '{label}' (atteso, applicato): {diff}")
    else:
        print(f"  Meteo verificato sul server: {label}")
    return not diff


# Test: ogni chiamata restituisce un oggetto nuovo, non influenzato da modifiche precedenti
_w = build_weather('fog')
_w.fog_density = 99.0
assert build_weather('fog').fog_density == 45.0
assert build_weather('backlight').fog_density == 0.0
assert weather_params('backlight', ego_yaw=-90.0, spawn_index=103)['sun_azimuth_angle'] == (backlight_offset(103) - 90.0) % 360.0
assert weather_params('backlight', ego_yaw=-90.0, spawn_index=0)['sun_azimuth_angle'] == (backlight_offset(0) - 90.0) % 360.0
for _name in WEATHER_OVERRIDES:
    _got = weather_to_dict(build_weather(_name))
    assert all(abs(_got[k] - v) <= 1e-3 for k, v in weather_params(_name).items()), _name


# ============================================================================
# -- Cella 11 del notebook ------------------------------------------------------
# ============================================================================

class MovingScenario:
    """
    Scenario controllato: ego vehicle si muove verso un pedone che attraversa (PED_CROSSING)
    oppure fermo davanti a lui.

    Attraversamento: il pedone parte dallo spartitraffico (o marciapiede) piu' vicino alla
    sinistra dell'ego, sulla linea perpendicolare alla marcia che passa per il punto a
    ped_distance metri dallo spawn (dove stava il pedone fermo), e cammina verso destra.
    Parte PED_START_DELAY_S dopo l'ego, a una velocita' (calcolata in run_condition) che lo porta
    davanti all'angolo anteriore destro dell'ego quando il paraurti arriva sulla sua linea: incidente.

    Contorno: i veicoli di OPPOSITE_TRAFFIC incrociano l'ego nella carreggiata opposta durante
    l'avvicinamento, le persone di SIDEWALK_WALKERS camminano sul marciapiede a destra. Le loro
    detection 'person' non sono errori (other_road_users, contate in other_person_detections).
    """
    SPAWN_INDEX = 0

    def __init__(self, world):
        self.world = world
        self.vehicles = []
        self.pedestrians = []
        self.sensors = []
        self.blueprint_lib = world.get_blueprint_library()

    def setup(self, ped_distance=80):
        """
        Crea scenario con pedone a distanza fissa.
        
        Args:
            ped_distance (int): distanza iniziale pedone in metri
        """
        spawn_points = self.world.get_map().get_spawn_points()
        ego_spawn = spawn_points[self.SPAWN_INDEX]

        # Spawn ego vehicle
        vehicle_bp = self.blueprint_lib.find('vehicle.tesla.model3')
        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, ego_spawn)
        self.vehicles.append(self.ego_vehicle)

        tesla_lights = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam
        self.ego_vehicle.set_light_state(carla.VehicleLightState(tesla_lights))
        
        # Punto davanti al veicolo, lungo la sua direzione di marcia: il pedone sta qui (fermo,
        # rivolto verso l'ego) oppure lo attraversa camminando verso destra
        ped_location = ego_spawn.location + ego_spawn.get_forward_vector() * ped_distance
        right = ego_spawn.get_right_vector()
        self.cross_line = ped_location
        self.cross_dir = carla.Vector3D(right.x, right.y, 0.0)
        pedestrian_bp = self.blueprint_lib.find('walker.pedestrian.0001')
        if PED_CROSSING:
            start_location = self._crossing_start(ped_location, right)
            ped_transform = carla.Transform(start_location, carla.Rotation(0, ego_spawn.rotation.yaw + 90.0, 0))
        else:
            ped_transform = carla.Transform(ped_location, carla.Rotation(0, ego_spawn.rotation.yaw + 180.0, 0))
        pedestrian = self.world.spawn_actor(pedestrian_bp, ped_transform)
        self.pedestrians.append(pedestrian)
        
        # Traffico nella carreggiata opposta e persone sul marciapiede a destra (stessi in ogni run)
        self.ego_spawn = ego_spawn
        self.opp_traffic = []
        self.lead_traffic = []
        self._plan_opposite_traffic(ego_spawn, ped_location)
        self._spawn_lead_traffic(ego_spawn)
        self._spawn_sidewalk_walkers(ego_spawn)
        return self.ego_vehicle, self.pedestrians

    def _rel(self, ego_spawn, location):
        """(distanza in avanti, spostamento a destra) di location rispetto allo spawn dell'ego."""
        fwd, right = ego_spawn.get_forward_vector(), ego_spawn.get_right_vector()
        d = location - ego_spawn.location
        return d.x * fwd.x + d.y * fwd.y, d.x * right.x + d.y * right.y

    def _opposite_lane(self, ego_spawn, ped_location):
        """Waypoint della corsia opposta (senso di marcia contrario all'ego) sulla linea del pedone."""
        carla_map = self.world.get_map()
        ped_wp = carla_map.get_waypoint(ped_location)
        right = ped_wp.transform.get_right_vector()
        for offset in (-6.0, -10.0, 6.0, 10.0):
            wp = carla_map.get_waypoint(ped_location + right * offset, project_to_road=True,
                                        lane_type=carla.LaneType.Driving)
            if wp is None or (wp.road_id == ped_wp.road_id and wp.lane_id == ped_wp.lane_id):
                continue
            yaw_diff = abs((wp.transform.rotation.yaw - ego_spawn.rotation.yaw + 180.0) % 360.0 - 180.0)
            if yaw_diff > 150.0:
                return wp
        return None

    def _plan_opposite_traffic(self, ego_spawn, ped_location):
        """
        Veicoli di OPPOSITE_TRAFFIC nella corsia opposta, fermi alla distanza indicata davanti allo
        spawn; partono al tick 1 insieme all'ego (update_traffic). Se un incrocio accorcia la corsia,
        si parte dal punto piu' lontano raggiungibile, in fila ogni 12 m.
        """
        if not OPPOSITE_TRAFFIC:
            return
        lane_wp = self._opposite_lane(ego_spawn, ped_location)
        if lane_wp is None:
            print("  AVVISO: corsia opposta non trovata, niente traffico")
            return

        # Punto piu' lontano raggiungibile lungo la corsia opposta senza entrare in un incrocio
        # (la corsia va verso l'ego: previous() si allontana)
        far_wp = lane_wp
        for _ in range(300):
            prev = far_wp.previous(1.0)
            if not prev or prev[0].is_junction:
                break
            far_wp = prev[0]
        limit = self._rel(ego_spawn, far_wp.transform.location)[0]
        starts = []
        for name, kmh, ahead_m in reversed(OPPOSITE_TRAFFIC):
            starts.insert(0, min(ahead_m, limit))
            limit = starts[0] - 12.0

        for (name, kmh, ahead_m), x_start in zip(OPPOSITE_TRAFFIC, starts):
            # A passi di 1 m lungo la corsia fino a meno di 1 m da x_start; si ferma se un passo
            # non avvicina (curva, fine corsia)
            wp = lane_wp
            x = self._rel(ego_spawn, wp.transform.location)[0]
            for _ in range(400):
                diff = x_start - x
                if abs(diff) < 1.0:
                    break
                nxt = wp.previous(1.0) if diff > 0 else wp.next(1.0)
                if not nxt:
                    break
                new_x = self._rel(ego_spawn, nxt[0].transform.location)[0]
                if abs(x_start - new_x) >= abs(diff):
                    break
                wp, x = nxt[0], new_x
            tf = carla.Transform(wp.transform.location + carla.Location(z=0.5), wp.transform.rotation)
            blueprints = self.blueprint_lib.filter(name) or self.blueprint_lib.filter('vehicle.audi.etron')
            vehicle = self.world.try_spawn_actor(blueprints[0], tf)
            if vehicle is None:
                print(f"  AVVISO: spawn fallito per {name} nella corsia opposta")
                continue
            vehicle.set_light_state(carla.VehicleLightState(
                carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam))
            vehicle.apply_control(carla.VehicleControl(brake=1.0))
            self.vehicles.append(vehicle)
            self.opp_traffic.append({'actor': vehicle, 'speed': kmh / 3.6, 'current': None})
            print(f"  Traffico opposto: {vehicle.type_id} parte da {x:.0f} m davanti all'ego, fino a {kmh:.0f} km/h")

    def _spawn_lead_traffic(self, ego_spawn):
        """
        Veicoli di LEAD_TRAFFIC nella corsia a destra dell'ego (stesso senso di marcia), alla distanza
        indicata davanti allo spawn. Per ognuno si cerca l'inizio dell'incrocio in fondo alla corsia:
        update_traffic lo fa fermare LEAD_STOP_MARGIN_M prima.
        """
        carla_map = self.world.get_map()
        fwd = ego_spawn.get_forward_vector()
        for name, kmh, ahead_m in LEAD_TRAFFIC:
            wp = carla_map.get_waypoint(ego_spawn.location + fwd * ahead_m).get_right_lane()
            if wp is None or wp.lane_type != carla.LaneType.Driving or \
                    abs((wp.transform.rotation.yaw - ego_spawn.rotation.yaw + 180.0) % 360.0 - 180.0) > 30.0:
                print(f"  AVVISO: corsia di destra (stesso senso dell'ego) non trovata a {ahead_m:.0f} m")
                continue
            # Inizio dell'incrocio in fondo alla corsia
            end_wp = wp
            for _ in range(300):
                nxt = end_wp.next(1.0)
                if not nxt or nxt[0].is_junction:
                    break
                end_wp = nxt[0]
            stop_x = self._rel(ego_spawn, end_wp.transform.location)[0] - LEAD_STOP_MARGIN_M
            tf = carla.Transform(wp.transform.location + carla.Location(z=0.5), wp.transform.rotation)
            blueprints = self.blueprint_lib.filter(name) or self.blueprint_lib.filter('vehicle.audi.etron')
            vehicle = self.world.try_spawn_actor(blueprints[0], tf)
            if vehicle is None:
                print(f"  AVVISO: spawn fallito per {name} nella corsia di destra")
                continue
            vehicle.set_light_state(carla.VehicleLightState(
                carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam))
            vehicle.apply_control(carla.VehicleControl(brake=1.0))
            self.vehicles.append(vehicle)
            self.lead_traffic.append({'actor': vehicle, 'speed': kmh / 3.6, 'current': None,
                                      'stop_x': stop_x, 'stopped': False})
            room = stop_x + LEAD_STOP_MARGIN_M - (ahead_m + vehicle.bounding_box.extent.x)
            print(f"  Corsia di destra: {vehicle.type_id} parte da {ahead_m:.0f} m a {kmh:.0f} km/h; "
                  f"incrocio a {room:.0f} m dal suo muso" + (", si ferma prima" if LEAD_STOP_AT_JUNCTION else ""))

    def _spawn_sidewalk_walkers(self, ego_spawn):
        """
        Persone di SIDEWALK_WALKERS sul marciapiede alla destra dell'ego: dalla corsia dell'ego si va
        verso destra (get_right_lane) oltre la corsia libera, fino alla prima corsia di tipo Sidewalk.
        Due file a +-SIDEWALK_LANE_OFFSET_M dal centro del marciapiede, una per verso di marcia.
        """
        carla_map = self.world.get_map()
        fwd = ego_spawn.get_forward_vector()
        walker_bps = sorted(self.blueprint_lib.filter('walker.pedestrian.*'), key=lambda b: b.id)
        walker_bps = [b for b in walker_bps if b.id != 'walker.pedestrian.0001'] or walker_bps
        self.sidewalk_walkers = []
        for k, (ahead_m, direction, speed) in enumerate(SIDEWALK_WALKERS):
            wp = carla_map.get_waypoint(ego_spawn.location + fwd * ahead_m)
            crossed = []
            for _ in range(6):
                wp = wp.get_right_lane()
                if wp is None or wp.lane_type == carla.LaneType.Sidewalk:
                    break
                crossed.append(str(wp.lane_type))
            if wp is None or wp.lane_type != carla.LaneType.Sidewalk:
                print(f"  AVVISO: marciapiede a destra non trovato a {ahead_m:.0f} m (corsie: {crossed})")
                continue
            # due file: verso la strada (-) chi va come l'ego, verso l'interno (+) chi va verso l'ego
            right = ego_spawn.get_right_vector()
            side = -direction * SIDEWALK_LANE_OFFSET_M
            loc = wp.transform.location + carla.Location(x=right.x * side, y=right.y * side, z=1.0)
            yaw = ego_spawn.rotation.yaw + (0.0 if direction > 0 else 180.0)
            walker = self.world.try_spawn_actor(walker_bps[(3 * k + 1) % len(walker_bps)],
                                                carla.Transform(loc, carla.Rotation(yaw=yaw)))
            if walker is None:
                print(f"  AVVISO: spawn fallito per la persona sul marciapiede a {ahead_m:.0f} m")
                continue
            self.pedestrians.append(walker)
            walk_dir = carla.Vector3D(fwd.x * direction, fwd.y * direction, 0.0)
            self.sidewalk_walkers.append({'actor': walker, 'dir': walk_dir, 'speed': speed,
                                          'yaw': yaw, 'base': None})
            if k == 0:
                print(f"  Marciapiede: dopo le corsie {crossed}, a {self._rel(ego_spawn, loc)[1]:.1f} m a destra")
        print(f"  Marciapiede: {len(self.sidewalk_walkers)} persone")

    def other_road_users(self):
        """
        Attori che YOLO puo' rilevare come 'person' senza che sia un errore: le persone sul
        marciapiede e i veicoli a due ruote (il conducente fa parte del box del veicolo).
        """
        others = [w['actor'] for w in getattr(self, 'sidewalk_walkers', [])]
        others += [c['actor'] for c in self.opp_traffic
                   if c['actor'].attributes.get('number_of_wheels') == '2']
        return others

    def update_traffic(self, n):
        """
        Da chiamare prima del tick n (n = 1, 2, ...): i veicoli della corsia opposta partono al tick 1
        e accelerano con la rampa dell'ego (EGO_ACCEL_MPS2) fino alla propria velocita'. La chiamata
        al server parte solo se la velocita' cambia.
        """
        for car in self.opp_traffic:
            speed = min(car['speed'], EGO_ACCEL_MPS2 * n * FIXED_DELTA_S)
            if speed == car['current']:
                continue
            if car['current'] is None:
                car['actor'].apply_control(carla.VehicleControl(brake=0.0))
            car['actor'].enable_constant_velocity(carla.Vector3D(speed, 0.0, 0.0))
            car['current'] = speed
        # Corsia di destra: rampa da fermo e velocita' costante; con LEAD_STOP_AT_JUNCTION frenata dolce
        # fino a fermarsi prima dell'incrocio (v = sqrt(2 * decelerazione * spazio rimasto))
        for car in self.lead_traffic:
            if car['stopped']:
                continue
            front = self._rel(self.ego_spawn, car['actor'].get_location())[0] + car['actor'].bounding_box.extent.x
            speed = min(car['speed'], EGO_ACCEL_MPS2 * n * FIXED_DELTA_S)
            if LEAD_STOP_AT_JUNCTION:
                room = max(0.0, car['stop_x'] - front)
                speed = min(speed, (2.0 * LEAD_DECEL_MPS2 * room) ** 0.5)
            if speed < 0.2 and n > 1:
                car['actor'].disable_constant_velocity()
                car['actor'].apply_control(carla.VehicleControl(brake=1.0))
                car['stopped'] = True
                continue
            if car['current'] is None:
                car['actor'].apply_control(carla.VehicleControl(brake=0.0))
            if abs(speed - (car['current'] or 0.0)) > 1e-3:
                car['actor'].enable_constant_velocity(carla.Vector3D(speed, 0.0, 0.0))
                car['current'] = speed
        # Persone sul marciapiede: prima del tick n stanno nel punto raggiunto dopo n-1 tick lungo la
        # loro retta; durante il tick il WalkerControl le fa avanzare (e anima la camminata)
        for w in getattr(self, 'sidewalk_walkers', []):
            s = w['speed'] * (n - 1) * FIXED_DELTA_S
            loc = carla.Location(w['base'].x + w['dir'].x * s, w['base'].y + w['dir'].y * s, w['base'].z)
            w['actor'].set_transform(carla.Transform(loc, carla.Rotation(yaw=w['yaw'])))

    def _crossing_start(self, line_point, right, max_offset_m=20.0, step_m=0.25, margin_m=0.5):
        """
        Punto di partenza dell'attraversamento: margin_m a destra del bordo dello spartitraffico
        (o marciapiede) piu' vicino alla sinistra della corsia dell'ego, sulla linea di
        attraversamento. Il pedone parte dal lato strada: le catene sul bordo dello
        spartitraffico gli impedirebbero di scendere dal cordolo.

        Returns:
            carla.Location di partenza
        """
        carla_map = self.world.get_map()
        lane_wp = carla_map.get_waypoint(line_point)
        d = lane_wp.transform.location - line_point
        lane_center = d.x * right.x + d.y * right.y
        offset = lane_center - lane_wp.lane_width / 2.0
        while offset > lane_center - max_offset_m:
            offset -= step_m
            wp = carla_map.get_waypoint(line_point + right * offset, project_to_road=False,
                                        lane_type=carla.LaneType.Any)
            if wp is not None and wp.lane_type in (carla.LaneType.Median, carla.LaneType.Sidewalk):
                # offset = primo punto (a step_m di precisione) oltre il bordo dello spartitraffico
                start_offset = offset + step_m + margin_m
                start = line_point + right * start_offset
                start.z = lane_wp.transform.location.z + 1.0   # sopra l'asfalto: il pedone si appoggia a terra
                print(f"  Pedone: parte accanto a {wp.lane_type}")
                return start
        raise RuntimeError("nessuno spartitraffico o marciapiede a sinistra della corsia dell'ego")

    def start_crossing(self, speed_mps):
        """Il pedone inizia a camminare verso destra a velocita' costante."""
        self.pedestrians[0].apply_control(carla.WalkerControl(direction=self.cross_dir, speed=speed_mps))

    def set_weather(self, weather_type):
        """Applica condizione meteo (backlight orientato sullo yaw dell'ego) e restituisce i parametri."""
        self.weather_params = weather_params(weather_type, ego_yaw=self.ego_vehicle.get_transform().rotation.yaw,
                                             spawn_index=self.SPAWN_INDEX)
        self.world.set_weather(carla.WeatherParameters(**self.weather_params))
        self.weather_params['street_lights'] = set_street_lights(self.world, weather_type, self.weather_params)
        self.weather_params['ego_lights'] = set_ego_lights(self.ego_vehicle, weather_type)
        return self.weather_params

    def start_motion(self):
        """
        Rilascia i freni dell'ego dopo l'assestamento (da qui la sua velocita' e' imposta con
        set_speed()) e fa partire le persone sul marciapiede. Il traffico opposto parte con update_traffic().
        """
        self._speed = None
        self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0))
        for w in getattr(self, 'sidewalk_walkers', []):
            w['base'] = w['actor'].get_location()   # appoggiato a terra dopo l'assestamento
            w['actor'].apply_control(carla.WalkerControl(direction=w['dir'], speed=w['speed']))

    def set_speed(self, speed_mps):
        """
        Velocita' imposta all'ego (enable_constant_velocity, coordinate locali); il traffico
        opposto ha velocita' proprie (update_traffic). Chiamata a ogni frame con la rampa di
        ramp_speed(): la chiamata al server parte solo se il valore cambia.
        """
        if speed_mps == self._speed:
            return
        self.ego_vehicle.enable_constant_velocity(carla.Vector3D(speed_mps, 0.0, 0.0))
        self._speed = speed_mps

    def cleanup(self):
        """Distruggi tutti gli attori creati e assicurati che spariscano."""
        for actor in self.sensors + self.vehicles + self.pedestrians:
            if actor.is_alive:
                actor.destroy()
        # Fai un tick sincrono per essere certi che il server li rimuova fisicamente
        # prima del prossimo test, evitando collisioni fantasma!
        try:
            settings = self.world.get_settings()
            was_sync = settings.synchronous_mode
            settings.synchronous_mode = True
            self.world.apply_settings(settings)
            self.world.tick()
            settings.synchronous_mode = was_sync
            self.world.apply_settings(settings)
        except:
            pass


# ============================================================================
# -- Cella 17 del notebook ------------------------------------------------------
# ============================================================================

def frame_to_rgb(image):
    """Converte immagine CARLA in array RGB numpy."""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    return array[:, :, :3][:, :, ::-1].copy()  # .copy() CRUCIALE: evita sovrascrittura buffer C++ CARLA!

# CORRUZIONI VISIVE — Funzioni per degradare le immagini
# I parametri in pixel (raggio blur, kernel motion blur) sono definiti per una larghezza
# di riferimento di 640 px e scalati con la risoluzione reale: la stessa severita'
# produce la stessa sfocatura RELATIVA a qualunque risoluzione.
CORRUPTION_REF_WIDTH = 640


def _resolution_scale(rgb_frame):
    return rgb_frame.shape[1] / CORRUPTION_REF_WIDTH

def apply_blur(rgb_frame, severity):
    """
    Applica Gaussian blur al frame.
    severity (sigma sulla larghezza di riferimento di 640 px; a 1280 px e' il doppio):
        1 = 0.5 (1 px), 2 = 1.0 (2 px), 3 = 1.5 (3 px, mild), 4 = 3.5 (7 px, severe)
    """
    if severity <= 0:
        return rgb_frame
    radius_map = {1: 0.5, 2: 1.0, 3: 1.5, 4: 3.5}
    radius = radius_map.get(severity, 3.5) * _resolution_scale(rgb_frame)
    # OpenCV: ~5-9x piu' veloce di PIL a 1280x720, risultato equivalente (sigma = radius)
    return cv2.GaussianBlur(rgb_frame, (0, 0), sigmaX=radius, sigmaY=radius)


def apply_noise(rgb_frame, severity):
    """
    Applica Gaussian noise al frame.
    severity: 1=mild (sigma=15), 2=moderate (sigma=30), 3=severe (sigma=50)
    """
    if severity <= 0:
        return rgb_frame
    sigma_map = {1: 15, 2: 30, 3: 50}
    sigma = sigma_map.get(severity, 30)
    noise = np.random.normal(0, sigma, rgb_frame.shape).astype(np.float32)
    noisy = np.clip(rgb_frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def apply_motion_blur(rgb_frame, severity):
    """
    Applica motion blur orizzontale (simula vibrazioni camera).
    severity: 1=mild (k=3), 2=moderate (k=7)
    """
    if severity <= 0:
        return rgb_frame
    kernel_size_map = {1: 3, 2: 7}
    k = kernel_size_map.get(severity, 5)
    k = max(1, int(round(k * _resolution_scale(rgb_frame))) | 1)  # dispari
    
    # Kernel di motion blur orizzontale (una riga di k pesi uguali)
    # NB: PIL.ImageFilter.Kernel accetta solo kernel 3x3 o 5x5, per questo si usa OpenCV
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(rgb_frame, -1, kernel)


def apply_contrast_reduction(rgb_frame, severity):
    """
    Riduce il contrasto dell'immagine.
    severity: 1=mild (0.7), 2=moderate (0.4)
    """
    if severity <= 0:
        return rgb_frame
    factor_map = {1: 0.7, 2: 0.4}
    factor = factor_map.get(severity, 0.4)
    pil_image = Image.fromarray(rgb_frame)
    enhancer = ImageEnhance.Contrast(pil_image)
    reduced = enhancer.enhance(factor)
    return np.array(reduced)


def apply_corruption(rgb_frame, corruption_type, severity):
    """
    Dispatcher: applica la corruzione specificata al frame.
    
    Args:
        rgb_frame: numpy array RGB
        corruption_type: 'none', 'blur', 'noise', 'motion_blur', 'contrast'
        severity: livello di severita (0 = nessuna)
    
    Returns:
        numpy array RGB con corruzione applicata
    """
    if corruption_type == 'none' or severity <= 0:
        return rgb_frame
    
    corruption_functions = {
        'blur': apply_blur,
        'noise': apply_noise,
        'motion_blur': apply_motion_blur,
        'contrast': apply_contrast_reduction,
    }
    
    fn = corruption_functions.get(corruption_type)
    if fn is None:
        raise ValueError(f"Corruzione sconosciuta: {corruption_type}")
    
    return fn(rgb_frame, severity)


def append_csv_rows(path, fieldnames, rows):
    """Aggiunge righe a CSV (crea file se non esiste)."""
    if not rows:
        return
    
    file_exists = os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)



# ── Spectator Chase-Cam (terza persona, stile videogioco) ──────────
def move_spectator_to(transform, spectator,
                      distance=7.0, x=0, y=0, z=3.0,
                      yaw=0, pitch=-15, roll=0):
    """
    Posiziona lo spectator di CARLA dietro al veicolo.
    Modifica direttamente il transform del veicolo per massima fluidita.
    """
    back_location = transform.location - transform.get_forward_vector() * distance

    back_location.x += x
    back_location.y += y
    back_location.z += z
    transform.rotation.yaw += yaw
    transform.rotation.pitch = pitch
    transform.rotation.roll = roll
    spectator_transform = carla.Transform(back_location, transform.rotation)
    spectator.set_transform(spectator_transform)




def yolo_input(rgb_frame):
    """Ultralytics interpreta gli array numpy come BGR (convenzione OpenCV): convertiamo da RGB."""
    return np.ascontiguousarray(rgb_frame[:, :, ::-1])


def spawn_rgb_camera(world, parent):
    """Camera RGB frontale (risoluzione, FOV e montaggio dalla configurazione, Sezione 7)."""
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(WIDTH))
    camera_bp.set_attribute('image_size_y', str(HEIGHT))
    camera_bp.set_attribute('fov', str(FOV))
    mount = carla.Transform(carla.Location(x=CAMERA_MOUNT[0], y=CAMERA_MOUNT[1], z=CAMERA_MOUNT[2]))
    return world.spawn_actor(camera_bp, mount, attach_to=parent)


def enable_sync_mode(world):
    """Modalita' sincrona a passo fisso FIXED_DELTA_S. Restituisce le impostazioni precedenti."""
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DELTA_S
    world.apply_settings(settings)
    return original


def ramp_speed(frame_index):
    """
    Velocita' imposta al frame k (k = 1, 2, ...): rampa dolce a EGO_ACCEL_MPS2 partendo
    da fermo (niente scatti che facciano sbandare l'auto), poi costante a EGO_SPEED_MPS.
    """
    return min(EGO_SPEED_MPS, EGO_ACCEL_MPS2 * frame_index * FIXED_DELTA_S)


def time_to_travel(distance_m):
    """Tempo simulato (s) che l'ego impiega a percorrere distance_m partendo da fermo con ramp_speed()."""
    travelled, k = 0.0, 0
    while travelled < distance_m:
        k += 1
        travelled += ramp_speed(k) * FIXED_DELTA_S
    return k * FIXED_DELTA_S


class DriftTracker:
    """Deriva dell'ego rispetto alla retta di partenza: serve a verificare che l'auto non sbandi."""

    def __init__(self, start_transform):
        self.origin = start_transform.location
        self.right = start_transform.get_right_vector()
        self.yaw0 = start_transform.rotation.yaw

    def __call__(self, transform):
        """Restituisce (spostamento laterale in m, variazione di direzione in gradi)."""
        d = transform.location - self.origin
        lateral = d.x * self.right.x + d.y * self.right.y
        heading_change = (transform.rotation.yaw - self.yaw0 + 180.0) % 360.0 - 180.0
        return lateral, heading_change


def select_device():
    """Sceglie l'accelerazione hardware disponibile per YOLO."""
    if torch.cuda.is_available():
        return 'cuda:0'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def camera_transform(vehicle):
    """Transform (coordinate mondo) che avrebbe la camera montata in CAMERA_MOUNT sul veicolo."""
    vehicle_tf = vehicle.get_transform()
    mount = carla.Location(x=CAMERA_MOUNT[0], y=CAMERA_MOUNT[1], z=CAMERA_MOUNT[2])
    location = vehicle_tf.transform(mount)
    return carla.Transform(location if location is not None else mount, vehicle_tf.rotation)


_last_tick_time = [0.0]


def tick_and_get_frame(world, image_queue, vehicle=None, timeout=5.0):
    """
    Avanza la simulazione di un tick (modalita' sincrona) e restituisce
    (frame RGB, snapshot, transform della camera) dello STESSO tick: niente
    immagini in ritardo. Le immagini di tick precedenti rimaste in coda vengono scartate.

    Senza camera (image_queue None, opzione --senza-camera) il frame e' None e il
    transform della camera e' calcolato dal montaggio sul veicolo; ogni tick dura almeno
    FIXED_DELTA_S di tempo reale, cosi' la simulazione si vede a velocita' naturale
    (i dati sono in tempo simulato e non cambiano).
    """
    if image_queue is None:
        wait = FIXED_DELTA_S - (time.time() - _last_tick_time[0])
        if wait > 0:
            time.sleep(wait)
        _last_tick_time[0] = time.time()
    frame_id = world.tick()
    if image_queue is None:
        return None, world.get_snapshot(), camera_transform(vehicle)
    while True:
        image = image_queue.get(timeout=timeout)
        if image.frame >= frame_id:
            return frame_to_rgb(image), world.get_snapshot(), image.transform


def get_annotated_rgb(result):
    """Genera e congela immediatamente una copia RGB del frame annotato da YOLO."""
    annotated = result.plot()
    return annotated[:, :, ::-1].copy()


def save_annotated_sample(sample_data, output_path):
    """Salva frame con bounding box YOLO (accetta sia Result YOLO che array numpy RGB congelato)."""
    if hasattr(sample_data, 'plot'):
        annotated = sample_data.plot()
        annotated_rgb = annotated[:, :, ::-1]
    else:
        annotated_rgb = sample_data
    Image.fromarray(annotated_rgb).save(output_path)


# ============================================================================
# -- Cella 19 del notebook ------------------------------------------------------
# ============================================================================

# GROUND TRUTH: proiezione dei bounding box 3D di CARLA e associazione con le detection
# (le soglie CONF_THRESHOLD, IOU_MATCH_THRESHOLD, MIN_GT_HEIGHT_PX, MIN_GT_VISIBLE_FRAC
#  sono definite nella cella di configurazione, Sezione 7)

GT_COLOR = (0, 255, 0)  # verde (RGB)


def camera_intrinsics(width, height, fov_deg):
    """Matrice K della camera pinhole di CARLA (pixel quadrati, centro ottico al centro)."""
    f = width / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    return np.array([[f, 0.0, width / 2.0],
                     [0.0, f, height / 2.0],
                     [0.0, 0.0, 1.0]])


def vertices_to_array(world_vertices):
    """Lista di carla.Location -> array numpy (N, 3)."""
    return np.array([[v.x, v.y, v.z] for v in world_vertices])


def project_box_2d(world_vertices, cam_transform, K, width, height):
    """
    Proietta nell'immagine i vertici (coordinate mondo) di un box 3D.

    Args:
        world_vertices: lista di carla.Location oppure array numpy (N, 3)
        cam_transform: carla.Transform della camera nel momento dello scatto
        K: matrice dei parametri intrinseci (camera_intrinsics)

    Returns:
        (x1, y1, x2, y2) ritagliato ai bordi dell'immagine, oppure None se il box
        e' dietro la camera, se meno di MIN_GT_VISIBLE_FRAC della sua area cade
        nell'inquadratura o se e' piu' basso di MIN_GT_HEIGHT_PX pixel.
    """
    pts = world_vertices if isinstance(world_vertices, np.ndarray) else vertices_to_array(world_vertices)
    pts_h = np.hstack([pts, np.ones((len(pts), 1))]).T
    world_to_cam = np.array(cam_transform.get_inverse_matrix())
    x_fwd, y_right, z_up = (world_to_cam @ pts_h)[:3]
    if np.any(x_fwd <= 0.1):
        return None
    # Assi UE (x avanti, y destra, z alto) -> assi camera (x destra, y basso, z avanti)
    uvw = K @ np.vstack([y_right, -z_up, x_fwd])
    u, v = uvw[0] / uvw[2], uvw[1] / uvw[2]
    x1, y1, x2, y2 = u.min(), v.min(), u.max(), v.max()
    cx1, cy1 = max(x1, 0.0), max(y1, 0.0)
    cx2, cy2 = min(x2, float(width)), min(y2, float(height))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    if (cx2 - cx1) * (cy2 - cy1) < MIN_GT_VISIBLE_FRAC * (x2 - x1) * (y2 - y1):
        return None
    if cy2 - cy1 < MIN_GT_HEIGHT_PX:
        return None
    return (float(cx1), float(cy1), float(cx2), float(cy2))


def box_iou(a, b):
    """Intersection over Union tra due box (x1, y1, x2, y2)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def boxes_array(result):
    """
    Tutte le detection del frame come array numpy (N, 6): x1, y1, x2, y2, conf, cls.
    Un solo trasferimento GPU -> CPU per frame: leggere i box uno a uno (box.conf[0], ...)
    forza una sincronizzazione con la GPU per ogni valore, e con YOLO_MIN_CONF basso i box sono tanti.
    """
    return result.boxes.data.cpu().numpy()


def detections_of_class(boxes, names, class_name):
    """Detection di una classe (da boxes_array) come lista di (conf, (x1, y1, x2, y2))."""
    return [(float(b[4]), (float(b[0]), float(b[1]), float(b[2]), float(b[3])))
            for b in boxes if names[int(b[5])] == class_name]


def count_classes(boxes, names, counter, min_conf):
    """Aggiorna il conteggio per classe delle detection con confidenza >= min_conf (diagnostica)."""
    for b in boxes[boxes[:, 4] >= min_conf]:
        name = names[int(b[5])]
        counter[name] = counter.get(name, 0) + 1


def match_detections(detections, gt_boxes):
    """
    Associazione greedy detection <-> ground truth (schema COCO): le detection sono
    considerate in ordine di confidenza decrescente e ognuna prende il box GT libero
    con IoU piu' alto, se >= IOU_MATCH_THRESHOLD.

    Args:
        detections: lista di (conf, box)
        gt_boxes: dict {gt_id: box}

    Returns:
        matches: dict {gt_id: (conf, iou, box)}
        unmatched: lista di (conf, box) non associate a nessun GT
    """
    matches, unmatched = {}, []
    for conf, box in sorted(detections, key=lambda d: -d[0]):
        best_id, best_iou = None, IOU_MATCH_THRESHOLD
        for gt_id, gt_box in gt_boxes.items():
            if gt_id in matches:
                continue
            iou = box_iou(box, gt_box)
            if iou >= best_iou:
                best_id, best_iou = gt_id, iou
        if best_id is None:
            unmatched.append((conf, box))
        else:
            matches[best_id] = (conf, best_iou, box)
    return matches, unmatched


def ground_point(box, K, cam_height):
    """
    Stima la posizione a terra di un oggetto dal bordo inferiore del suo box
    (strada piana, camera orizzontale), come farebbe una FCW monoculare.

    Returns:
        (distanza in avanti, spostamento laterale) in metri rispetto alla camera,
        oppure None se il bordo inferiore del box e' sopra l'orizzonte.
    """
    f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    u, v = (box[0] + box[2]) / 2.0, box[3]
    if v <= cy + 1.0:
        return None
    forward = f * cam_height / (v - cy)
    return forward, (u - cx) * forward / f


def format_confs(confs):
    """Confidenze come stringa compatta per il CSV (es. '0.62;0.41')."""
    return ';'.join(f"{c:.3f}" for c in sorted(confs, reverse=True))


def draw_gt_box(image_rgb, box, label='GT'):
    """Disegna un box di ground truth in verde, con etichetta sotto il box."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(image_rgb, (x1, y1), (x2, y2), GT_COLOR, 1)
    cv2.putText(image_rgb, label, (x1, min(y2 + 12, image_rgb.shape[0] - 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, GT_COLOR, 1)
    return image_rgb


def gt_zoom(image, frame_row, scale=3):
    """Ritaglio ingrandito attorno al box di ground truth di un frame."""
    x1, y1, x2, y2 = (frame_row[k] for k in ('gt_x1', 'gt_y1', 'gt_x2', 'gt_y2'))
    margin = max(40.0, 1.5 * (y2 - y1))
    h, w = image.shape[:2]
    cx1, cy1 = int(max(0, x1 - margin)), int(max(0, y1 - margin))
    cx2, cy2 = int(min(w, x2 + margin)), int(min(h, y2 + margin))
    crop = image[cy1:cy2, cx1:cx2]
    return cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale), interpolation=cv2.INTER_NEAREST)


def annotate_frame(result, gt_boxes=()):
    """Frame annotato (RGB): detection YOLO sopra CONF_THRESHOLD + box di ground truth in verde."""
    if result is None:   # senza camera: nessuna immagine (il salvataggio dei sample viene saltato)
        return None
    shown = result[result.boxes.conf >= CONF_THRESHOLD]
    img = get_annotated_rgb(shown)
    for box in gt_boxes:
        draw_gt_box(img, box)
    return img


# ============================================================================
# -- Cella 21 del notebook ------------------------------------------------------
# ============================================================================

# Parametri simulazione
WIDTH, HEIGHT = 1280, 720     # a 640x360 un pedone a 40 m era alto ~14 px: troppo piccolo per YOLOv8n
FOV = 90
CAMERA_MOUNT = (1.5, 0.0, 2.4)  # posizione camera rispetto all'ego (x avanti, y destra, z alto), metri
FIXED_DELTA_S = 0.05          # passo di simulazione (20 FPS, modalita' sincrona)
YOLO_IMGSZ = WIDTH            # dimensione di input di YOLO: senza questo YOLO ridimensiona a 640
YOLO_HALF = True              # inferenza FP16 su GPU NVIDIA: piu' veloce, ignorato su CPU/MPS
PEDESTRIAN_DISTANCE_M = 80
WARNING_DISTANCE_M = 15.0
MAX_FRAMES_PER_RUN = 500

# Moto: velocita' COSTANTE per ego e traffico (stessa in ogni condizione)
EGO_SPEED_KMH = 40.0          # a 40 km/h il warning (TTC 2.5 s) va dato a ~28 m: li' la detection inizia a cedere
EGO_SPEED_MPS = EGO_SPEED_KMH / 3.6
EGO_ACCEL_MPS2 = 2.5          # rampa di partenza da fermo: ~4.4 s e ~25 m per arrivare a 40 km/h
PED_CROSSING = True           # Scenario 1: il pedone attraversa (dallo spartitraffico a sinistra verso destra)
                              # e arriva davanti all'angolo destro dell'ego quando il paraurti raggiunge la sua linea (incidente)
PED_START_DELAY_S = 5.0       # il pedone parte 5 s dopo l'ego: attraversa lentamente durante l'avvicinamento
# Scenario 1: traffico nella carreggiata opposta. Tutti partono insieme all'ego e accelerano con la
# stessa rampa (EGO_ACCEL_MPS2) fino alla propria velocita'. Chi parte piu' lontano va piu' piano,
# quindi nella stessa corsia nessuno raggiunge il veicolo davanti.
# (blueprint, velocita' km/h, distanza di partenza davanti allo spawn dell'ego in m)
OPPOSITE_TRAFFIC = [
    ('vehicle.audi.etron', 40.0, 90.0),
    ('vehicle.yamaha.yzf', 35.0, 110.0),
    ('vehicle.mini.cooper_s_2021', 30.0, 130.0),
]
# Scenario 1: veicoli nella corsia di destra (stesso senso dell'ego), oltre la linea del pedone: vanno piano
# verso il semaforo in fondo per tutto il run. Con LEAD_STOP_AT_JUNCTION = True si fermano invece
# LEAD_STOP_MARGIN_M prima dell'incrocio (decelerazione LEAD_DECEL_MPS2): serve spazio sufficiente.
# (blueprint, velocita' km/h, distanza di partenza davanti allo spawn dell'ego in m)
LEAD_TRAFFIC = [
    ('vehicle.nissan.patrol_2021', 20.0, 92.0),
]
LEAD_STOP_AT_JUNCTION = False
LEAD_STOP_MARGIN_M = 3.0
LEAD_DECEL_MPS2 = 3.0
# Scenario 1: persone sul marciapiede a destra dell'ego (non sono il target: detection contate a parte).
# Chi va come l'ego cammina SIDEWALK_LANE_OFFSET_M verso la strada, chi va verso l'ego verso l'interno:
# non si incontrano mai. Il percorso e' imposto a ogni tick (update_traffic): nessuno puo' essere spinto in strada.
# (distanza davanti allo spawn in m, verso: +1 = come l'ego, -1 = verso l'ego, velocita' m/s)
SIDEWALK_WALKERS = [
    (20.0, +1, 1.3),
    (38.0, -1, 1.1),
    (52.0, +1, 1.4),
    (66.0, -1, 1.2),
    (85.0, -1, 1.0),
]
SIDEWALK_LANE_OFFSET_M = 0.5
CROSS_TRAFFIC = True          # Scenario 2: traffico sulla strada trasversale (verde) e incidente dopo il rosso
CROSS_EARLY_GAP_M = 30.0      # auto da destra gia' nell'incrocio: passa davanti all'ego col paraurti a 30 m dalla linea
CROSS_FAST_KMH = 60.0         # auto veloci dai lati, fuori dall'inquadratura durante l'avvicinamento
CROSS_NEAR_MISS_S = 0.5       # auto veloce da destra: passa davanti all'ego 0.5 s dopo la linea d'arresto
CROSS_FAST_PARK_M = 60.0      # ...partendo da 60 m prima dell'incrocio
CROSS_KINEMATIC_BACK_M = 80.0 # auto veloce da sinistra (strada in curva): segue la corsia da 80 m prima dell'incrocio
TL_POST_CRASH_S = 2.0         # l'ego non si ferma al rosso: si simula fino a 2 s dopo l'urto
                              # (le metriche si fermano comunque alla linea d'arresto)
TL_SPAWN_BACK_M = 0.0         # Scenario 2: lo spawn 103 ha un incrocio subito dietro, non si puo' arretrare

# Warning (FCW per il pedone, red light warning per il semaforo), Sezione 8.1
FCW_TTC_WARNING_S = 2.5       # il warning va emesso quando il time-to-collision scende sotto questa soglia
FCW_CONFIRM_K, FCW_CONFIRM_N = 3, 5  # target confermato se rilevato in almeno K degli ultimi N frame
FCW_REACTION_S = 1.0          # tempo di reazione del guidatore dopo il warning
FCW_DECEL_MPS2 = 6.0          # decelerazione di frenata
EGO_PATH_HALF_WIDTH_M = 1.5   # semi-larghezza del percorso dell'ego (per i falsi allarmi FCW)

# Detection e ground truth (stessi valori per entrambi gli scenari)
CONF_THRESHOLD = 0.55       # confidenza minima perche' una detection conti (TP o FP): valore severo
YOLO_MIN_CONF = 0.10        # YOLO restituisce i box da questa confidenza: vengono salvati nel CSV
                            # cosi' le metriche si possono ricalcolare con un'altra soglia senza rieseguire
IOU_MATCH_THRESHOLD = 0.30  # IoU minimo per associare una detection alla ground truth
MIN_GT_HEIGHT_PX = 4        # target piu' bassi di cosi' (in pixel) sono considerati non visibili
MIN_GT_VISIBLE_FRAC = 0.5   # frazione minima del box di ground truth dentro l'inquadratura

# CONDIZIONI DA TESTARE (16): meteo base, meteo severo e intermedio, combinazioni, blur su 4 livelli
# Traccia: "clear weather, rain, fog, night, backlight, and blur"
#          "Add mild and severe visual corruptions to identify clear failure thresholds"

TESTS = [
    # --- Condizioni meteo (baseline: nessuna corruzione) ---
    {'weather': 'clear',     'corruption': 'none', 'severity': 0},
    {'weather': 'rain',      'corruption': 'none', 'severity': 0},
    {'weather': 'fog',       'corruption': 'none', 'severity': 0},
    {'weather': 'night',     'corruption': 'none', 'severity': 0},
    {'weather': 'backlight', 'corruption': 'none', 'severity': 0},

    # --- Meteo severo e livelli intermedi di nebbia (soglie di fallimento) ---
    {'weather': 'rain_severe',  'corruption': 'none', 'severity': 0},
    {'weather': 'fog_60',       'corruption': 'none', 'severity': 0},
    {'weather': 'fog_75',       'corruption': 'none', 'severity': 0},
    {'weather': 'fog_severe',   'corruption': 'none', 'severity': 0},
    {'weather': 'night_severe', 'corruption': 'none', 'severity': 0},

    # --- Combinazioni ---
    {'weather': 'night_highbeam', 'corruption': 'none', 'severity': 0},  # notte senza lampioni, abbaglianti
    {'weather': 'rain_night',     'corruption': 'none', 'severity': 0},  # pioggia di notte, lampioni accesi

    # --- Blur: sigma 1, 2, 3 (mild), 7 px (severe) a 1280 px ---
    {'weather': 'clear', 'corruption': 'blur', 'severity': 1},
    {'weather': 'clear', 'corruption': 'blur', 'severity': 2},
    {'weather': 'clear', 'corruption': 'blur', 'severity': 3},  # mild
    {'weather': 'clear', 'corruption': 'blur', 'severity': 4},  # severe
]


def save_benchmark_config(path, scenario, summary_rows, client, world):
    """Salva in JSON la configurazione completa del batch insieme al meteo applicato in ogni run."""
    try:
        carla_info = {'client_version': client.get_client_version(),
                      'server_version': client.get_server_version(),
                      'map': world.get_map().name}
    except Exception as exc:
        carla_info = {'error': str(exc)}
    cfg = {
        'scenario': scenario,
        'created': datetime.now().isoformat(timespec='seconds'),
        'carla': carla_info,
        'camera': {'width': WIDTH, 'height': HEIGHT, 'fov': FOV, 'mount_xyz': list(CAMERA_MOUNT),
                   'fixed_delta_s': FIXED_DELTA_S},
        'model': {'path': globals().get('MODEL_PATH'), 'imgsz': YOLO_IMGSZ, 'yolo_min_conf': YOLO_MIN_CONF,
                  'half': YOLO_HALF},
        'detection': {'conf_threshold': CONF_THRESHOLD, 'iou_match_threshold': IOU_MATCH_THRESHOLD,
                      'min_gt_height_px': MIN_GT_HEIGHT_PX, 'min_gt_visible_frac': MIN_GT_VISIBLE_FRAC},
        'motion': {'ego_speed_kmh': EGO_SPEED_KMH, 'ego_accel_mps2': EGO_ACCEL_MPS2,
                   'pedestrian_distance_m': PEDESTRIAN_DISTANCE_M, 'tl_spawn_back_m': TL_SPAWN_BACK_M,
                   'pedestrian_crossing': PED_CROSSING, 'ped_start_delay_s': PED_START_DELAY_S,
                   'opposite_traffic': OPPOSITE_TRAFFIC, 'lead_traffic': LEAD_TRAFFIC,
                   'sidewalk_walkers': SIDEWALK_WALKERS,
                   'cross_traffic': CROSS_TRAFFIC, 'cross_early_gap_m': CROSS_EARLY_GAP_M,
                   'cross_fast_kmh': CROSS_FAST_KMH, 'cross_near_miss_s': CROSS_NEAR_MISS_S,
                   'tl_post_crash_s': TL_POST_CRASH_S,
                   'max_frames_per_run': MAX_FRAMES_PER_RUN},
        'warning': {'ttc_warning_s': FCW_TTC_WARNING_S, 'confirm_k': FCW_CONFIRM_K, 'confirm_n': FCW_CONFIRM_N,
                    'reaction_s': FCW_REACTION_S, 'decel_mps2': FCW_DECEL_MPS2,
                    'ego_path_half_width_m': EGO_PATH_HALF_WIDTH_M},
        'weather_base': BASE_WEATHER,
        'street_lights_off': sorted(STREET_LIGHTS_OFF),
        'ego_high_beam': sorted(EGO_HIGH_BEAM),
        'backlight_azimuth_offset': BACKLIGHT_AZIMUTH_OFFSET,
        'corruption_ref_width': CORRUPTION_REF_WIDTH,
        'tests': TESTS,
        'runs': [{'run_label': r['run_label'], 'status': r['status'],
                  'weather_params': json.loads(r['weather_params']) if r.get('weather_params') else None}
                 for r in summary_rows],
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"   Configurazione salvata: {path}")


# ============================================================================
# -- Cella 23 del notebook ------------------------------------------------------
# ============================================================================

def run_condition(world, model, batch_id, weather, corruption, severity, samples_dir, camera_enabled=True):
    """
    Esegue un test completo per una condizione specifica.
    
    Ogni detection 'person' viene associata (IoU) al box di ground truth del pedone
    proiettato da CARLA: TP = pedone visibile e rilevato, FN = visibile ma non rilevato,
    FP = detection 'person' che non corrisponde al pedone.

    Metriche raccolte:
    - detection_rate (= recall: TP / frame con pedone visibile), precision
    - first_detection_distance, min_distance
    - warning_delay_s (tempo simulato tra zona warning e prima detection)
    - miss_rate (frame in zona critica con pedone visibile ma non rilevato)
    - false_positive_count, fp_per_frame

    Returns:
        tuple: (summary_row, frame_rows)
    """
    run_label = f"{weather}_{corruption}{severity}"
    print(f"\n{'='*50}")
    print(f"  Running: {run_label}")
    
    print(f"{'='*50}")

    # Zona critica: distanza sotto cui il pedone DOVREBBE essere rilevato
    CRITICAL_DISTANCE_M = 30.0

    frame_rows = []
    summary_row = {
        'batch_id': batch_id,
        'run_label': run_label,
        'weather': weather,
        'corruption': corruption,
        'severity': severity,
        'total_frames': 0,
        'detection_frames': 0,
        'detection_rate': 0.0,
        'first_detection_distance_m': None,
        'first_detection_elapsed_s': None,
        'warning_delay_s': None,
        'min_distance_m': None,
        'miss_rate': 0.0,
        'gt_visible_frames': 0,
        'false_negatives': 0,
        'false_positive_count': 0,
        'precision': 0.0,
        'fp_per_frame': 0.0,
        'critical_zone_frames': 0,
        'critical_zone_misses': 0,
        'weather_params': None,
        'weather_ok': None,
        'status': 'ok',
    }

    scenario = None
    camera = None
    first_detection_distance = None
    first_detection_time = None
    entered_warning_time = None
    frame_count = 0
    detection_count = 0      # frame TP
    fn_count = 0             # frame FN
    gt_visible_frames = 0    # frame con pedone visibile (denominatore del detection rate)
    min_distance = None
    detection_classes = {}
    false_positive_count = 0
    critical_zone_frames = 0
    critical_zone_misses = 0
    
    # Sotto-cartella per sample di questa condizione
    condition_samples_dir = os.path.join(samples_dir, run_label)
    os.makedirs(condition_samples_dir, exist_ok=True)
    samples_saved = 0

    # Definiti prima del try: il blocco finally li usa anche se il setup fallisce
    original_settings = world.get_settings()  # per ripristinarlo alla fine
    device = select_device()
    use_half = YOLO_HALF and device.startswith('cuda')

    try:
        # Abilita Synchronous Mode (passo fisso FIXED_DELTA_S)
        enable_sync_mode(world)
        settings = world.get_settings()
        print(f"  Modalita' sincrona: {settings.synchronous_mode}, passo fisso {settings.fixed_delta_seconds} s")
        
        spectator = world.get_spectator()

        # Setup scenario
        scenario = MovingScenario(world)
        scenario.setup(ped_distance=PEDESTRIAN_DISTANCE_M)
        summary_row['weather_params'] = json.dumps(scenario.set_weather(weather))

        # Attach camera (senza camera: niente immagini ne' YOLO, solo moto e ground truth)
        image_queue = None
        if camera_enabled:
            camera = spawn_rgb_camera(world, scenario.ego_vehicle)

            # Coda: ogni immagine viene associata al tick che l'ha generata
            image_queue = queue.Queue()
            camera.listen(image_queue.put)
        scenario.ego_vehicle.set_autopilot(False)

        # --- SETTLING PHASE  ---
        # Facciamo scorrere la simulazione per 1 secondo (20 tick) a vuoto
        # per far stabilizzare le sospensioni della macchina (evita sbandamenti)
        
        for _ in range(10):
            scenario.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
            tick_and_get_frame(world, image_queue, scenario.ego_vehicle)
            move_spectator_to(scenario.ego_vehicle.get_transform(), spectator, distance=8.0, z=3.0, pitch=-20)
        # ---------------------------------------------------------

        summary_row['weather_ok'] = check_applied_weather(world, scenario.weather_params, weather)

        # Sensore di collisione: qualunque contatto con l'ego viene registrato (verifica dello scenario)
        collisions = []
        collision_sensor = world.spawn_actor(world.get_blueprint_library().find('sensor.other.collision'),
                                             carla.Transform(), attach_to=scenario.ego_vehicle)
        scenario.sensors.append(collision_sensor)
        collision_sensor.listen(lambda e: collisions.append((e.frame, e.other_actor.type_id, e.other_actor.id)))

        # Partenza: ego e traffico a velocita' costante
        scenario.start_motion()
        start_frame = world.get_snapshot().frame
        drift = DriftTracker(scenario.ego_vehicle.get_transform())
        sim_start_time = world.get_snapshot().timestamp.elapsed_seconds

        # Ground truth: parametri camera e bounding box 3D del pedone
        K = camera_intrinsics(WIDTH, HEIGHT, FOV)
        pedestrian = scenario.pedestrians[0]
        ped_bbox = pedestrian.bounding_box
        # Persone sul marciapiede e veicoli a due ruote: una detection 'person' su di loro non e' un errore
        others = [(a.id, a.bounding_box) for a in scenario.other_road_users()]
        other_person_count = 0
        ego_half_length = scenario.ego_vehicle.bounding_box.extent.x
        fwd0 = scenario.ego_vehicle.get_transform().get_forward_vector()

        # Attraversamento: il pedone parte PED_START_DELAY_S dopo l'ego, alla velocita' che lo porta
        # davanti all'angolo anteriore destro dell'ego quando il paraurti arriva sulla sua linea (incidente)
        if PED_CROSSING:
            ego_loc0, line = scenario.ego_vehicle.get_location(), scenario.cross_line
            gap0 = (line.x - ego_loc0.x) * fwd0.x + (line.y - ego_loc0.y) * fwd0.y - ego_half_length
            p0 = pedestrian.get_location()
            impact_lateral = scenario.ego_vehicle.bounding_box.extent.y   # lato destro dell'auto
            path_m = impact_lateral - ((p0.x - drift.origin.x) * drift.right.x + (p0.y - drift.origin.y) * drift.right.y)
            ped_start_frame = int(round(PED_START_DELAY_S / FIXED_DELTA_S))
            walk_time = time_to_travel(gap0) - ped_start_frame * FIXED_DELTA_S
            if walk_time <= 0:
                raise RuntimeError(f"PED_START_DELAY_S={PED_START_DELAY_S} troppo grande: l'ego arriva prima")
            cross_speed = path_m / walk_time
            summary_row['ped_cross_speed_mps'] = round(cross_speed, 3)
            print(f"  Pedone: parte dopo {PED_START_DELAY_S:.1f} s e cammina {path_m:.1f} m a {cross_speed:.2f} m/s "
                  f"fino al lato destro dell'ego")

        key_samples = {}

        # Main loop
        while frame_count < MAX_FRAMES_PER_RUN:
            # Velocita': rampa dolce da fermo, poi costante (stessa in ogni condizione)
            scenario.set_speed(ramp_speed(frame_count + 1))
            scenario.update_traffic(frame_count + 1)
            if PED_CROSSING and frame_count == ped_start_frame:
                scenario.start_crossing(cross_speed)
            # Tick + immagine dello STESSO tick
            current_frame, snapshot, cam_tf = tick_and_get_frame(world, image_queue, scenario.ego_vehicle)

            # Sposta spectator insieme al tick sincrono
            move_spectator_to(scenario.ego_vehicle.get_transform(), spectator, distance=8.0, z=3.0, pitch=-20)

            frame_count += 1

            # Applica corruzione visiva (dispatcher generico)
            processed_frame = apply_corruption(current_frame, corruption, severity) if current_frame is not None else None

            # Calcola distanza veicolo-pedone
            v_loc = scenario.ego_vehicle.get_location()
            p_loc = scenario.pedestrians[0].get_location()
            distance = v_loc.distance(p_loc)
            min_distance = distance if min_distance is None else min(min_distance, distance)

            # Pedone rispetto all'ego: distanza longitudinale paraurti -> linea del pedone e
            # spostamento laterale rispetto alla retta di marcia (positivo = a destra)
            ped_gap = (p_loc.x - v_loc.x) * fwd0.x + (p_loc.y - v_loc.y) * fwd0.y - ego_half_length
            ped_lateral = (p_loc.x - drift.origin.x) * drift.right.x + (p_loc.y - drift.origin.y) * drift.right.y

            # Velocita' e time-to-collision (gap = paraurti anteriore -> pedone; se il pedone
            # attraversa: paraurti -> linea di attraversamento, cioe' tempo per arrivare al pedone)
            vel = scenario.ego_vehicle.get_velocity()
            speed = float(np.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2))
            lateral_offset, heading_change = drift(scenario.ego_vehicle.get_transform())
            gap = max(0.0, ped_gap if PED_CROSSING else distance - ego_half_length)
            ttc = gap / speed if speed > 0.1 else None
            elapsed_s = snapshot.timestamp.elapsed_seconds - sim_start_time  # tempo SIMULATO

            # Tracking zona warning (< 15m)
            if distance <= WARNING_DISTANCE_M and entered_warning_time is None:
                entered_warning_time = elapsed_s

            # Ground truth: box 2D del pedone in questo frame (None se non visibile)
            ped_tf = snapshot.find(pedestrian.id).get_transform()
            gt_box = project_box_2d(ped_bbox.get_world_vertices(ped_tf), cam_tf, K, WIDTH, HEIGHT)
            gt_visible = gt_box is not None
            gt_others = {}
            for actor_id, bbox in others:
                box = project_box_2d(bbox.get_world_vertices(snapshot.find(actor_id).get_transform()),
                                     cam_tf, K, WIDTH, HEIGHT)
                if box is not None:
                    gt_others[('other', actor_id)] = box

            # YOLO detection (box da YOLO_MIN_CONF in su: la soglia si applica dopo)
            if processed_frame is not None:
                results = model(yolo_input(processed_frame), verbose=False, device=device,
                                conf=YOLO_MIN_CONF, imgsz=YOLO_IMGSZ, half=use_half)
                result = results[0]

                # Tutti i box in numpy con un solo trasferimento dalla GPU
                boxes = boxes_array(result)
            else:
                # Senza camera: nessuna detection
                result = None
                boxes = np.zeros((0, 6), dtype=np.float32)
            count_classes(boxes, model.names, detection_classes, CONF_THRESHOLD)

            # Associazione detection 'person' <-> pedone e altre persone reali
            persons = detections_of_class(boxes, model.names, 'person')
            gt_all = dict(gt_others)
            if gt_visible:
                gt_all['ped'] = gt_box
            matches, unmatched = match_detections(persons, gt_all)
            frame_other = sum(1 for k, m in matches.items() if k != 'ped' and m[0] >= CONF_THRESHOLD)
            other_person_count += frame_other
            target_conf, target_iou = matches['ped'][:2] if 'ped' in matches else (0.0, 0.0)
            detected = gt_visible and target_conf >= CONF_THRESHOLD   # TP
            missed = gt_visible and not detected                       # FN
            fp_confs = [c for c, _ in unmatched]
            frame_fp = sum(c >= CONF_THRESHOLD for c in fp_confs)
            persons_detected = sum(c >= CONF_THRESHOLD for c, _ in persons)

            # Falso positivo "nel percorso": una FCW monoculare frenerebbe per questo oggetto
            # (posizione stimata a terra dal bordo inferiore del box, TTC stimato <= soglia)
            cam_height = cam_tf.location.z - v_loc.z
            fp_in_path = False
            for c, box in unmatched:
                if c < CONF_THRESHOLD:
                    continue
                ground = ground_point(box, K, cam_height)
                if ground and abs(ground[1]) <= EGO_PATH_HALF_WIDTH_M and speed > 0.1 \
                        and ground[0] / speed <= FCW_TTC_WARNING_S:
                    fp_in_path = True

            # Error rate tracking
            false_positive_count += frame_fp
            if gt_visible:
                gt_visible_frames += 1
            if missed:
                fn_count += 1

            # Zona critica (< 30m) per miss rate: solo frame con pedone visibile
            in_critical_zone = distance <= CRITICAL_DISTANCE_M
            if in_critical_zone and gt_visible:
                critical_zone_frames += 1
                if missed:
                    critical_zone_misses += 1

            if detected:
                detection_count += 1
                if first_detection_distance is None:
                    first_detection_distance = distance
                    first_detection_time = elapsed_s
                    print(f"  Prima detection PERSON: {distance:.1f}m (conf={target_conf:.2f}, IoU={target_iou:.2f})")

            # Frame-level row per CSV
            frame_rows.append({
                'batch_id': batch_id,
                'run_label': run_label,
                'weather': weather,
                'corruption': corruption,
                'severity': severity,
                'frame_index': frame_count,
                'elapsed_s': round(elapsed_s, 3),
                'distance_m': round(distance, 3),
                'persons_detected': persons_detected,
                'max_confidence': round(target_conf, 4),  # confidenza della detection associata al pedone
                'detected': int(detected),
                'in_critical_zone': int(in_critical_zone),
                'gt_visible': int(gt_visible),
                'gt_x1': round(gt_box[0], 1) if gt_visible else None,
                'gt_y1': round(gt_box[1], 1) if gt_visible else None,
                'gt_x2': round(gt_box[2], 1) if gt_visible else None,
                'gt_y2': round(gt_box[3], 1) if gt_visible else None,
                'gt_height_px': round(gt_box[3] - gt_box[1], 1) if gt_visible else None,
                'target_iou': round(target_iou, 3),
                'tp': int(detected),
                'fn': int(missed),
                'fp': frame_fp,
                'fp_confs': format_confs(fp_confs),
                'fp_in_path': int(fp_in_path),
                'speed_mps': round(speed, 3),
                'lateral_offset_m': round(lateral_offset, 3),
                'heading_change_deg': round(heading_change, 3),
                'gap_m': round(gap, 3),
                'ttc_s': round(ttc, 3) if ttc is not None else None,
                'ped_lateral_m': round(ped_lateral, 3),
                'other_persons_visible': len(gt_others),
                'other_person_detections': frame_other,
            })

            # Cattura campioni visivi ordinati e congelati in RAM a distanze chiave (dall'inizio alla fine)
            # Box verde = ground truth del pedone
            gt_draw = [gt_box] if gt_visible else []
            if frame_count == 1 and '01_start' not in key_samples:
                key_samples['01_start'] = (annotate_frame(result, gt_draw), f"01_start_{distance:.1f}m")
            elif distance <= 60.0 and '02_approach_60m' not in key_samples:
                key_samples['02_approach_60m'] = (annotate_frame(result, gt_draw), f"02_approach_{distance:.1f}m")
            elif distance <= 40.0 and '03_approach_40m' not in key_samples:
                key_samples['03_approach_40m'] = (annotate_frame(result, gt_draw), f"03_approach_{distance:.1f}m")
            elif distance <= 20.0 and '04_critical_20m' not in key_samples:
                key_samples['04_critical_20m'] = (annotate_frame(result, gt_draw), f"04_critical_{distance:.1f}m")

            # Traccia primo rilevamento pedone ed eventuale miss in zona critica
            if detected and 'detected' not in key_samples:
                key_samples['detected'] = (annotate_frame(result, gt_draw), f"det_first_{distance:.1f}m_conf{target_conf:.2f}")
            elif in_critical_zone and missed and 'miss' not in key_samples:
                key_samples['miss'] = (annotate_frame(result, gt_draw), f"miss_crit_{distance:.1f}m")

            # Stop se troppo vicino (<3m)
            # (pedone che attraversa: stop quando il paraurti arriva sulla sua linea, l'ego gli passa accanto)
            if distance < 3.0 or (PED_CROSSING and ped_gap <= 0.5):
                key_samples['05_final'] = (annotate_frame(result, gt_draw), f"05_final_{distance:.1f}m")
                print(f'  Veicolo arrivato al pedone (distanza {distance:.1f} m, pedone a {ped_lateral:+.1f} m '
                      f'lateralmente), stop.')
                break

        # Salva su disco i pochi sample chiave (solo alla fine della corsa, in un lampo!)
        for s_type, (s_res, s_label) in key_samples.items():
            try:
                s_path = os.path.join(condition_samples_dir, f"{s_label}.png")
                save_annotated_sample(s_res, s_path)
                samples_saved += 1
            except Exception:
                pass

        # Calcola metriche finali
        # detection_rate = recall sui frame in cui il pedone e' visibile
        detection_rate = detection_count / gt_visible_frames if gt_visible_frames else 0.0
        miss_rate = critical_zone_misses / critical_zone_frames if critical_zone_frames else 0.0
        precision = (detection_count / (detection_count + false_positive_count)
                     if (detection_count + false_positive_count) else 0.0)
        fp_per_frame = false_positive_count / frame_count if frame_count else 0.0
        
        # Warning delay: tempo tra ingresso in zona warning e prima detection
        warning_delay = None
        if first_detection_time is not None and entered_warning_time is not None:
            warning_delay = round(max(0.0, first_detection_time - entered_warning_time), 3)
        
        first_detection_elapsed = None
        if first_detection_time is not None:
            first_detection_elapsed = round(first_detection_time, 3)
        
        summary_row.update({
            'total_frames': frame_count,
            'detection_frames': detection_count,
            'detection_rate': round(detection_rate, 4),
            'first_detection_distance_m': round(first_detection_distance, 3) if first_detection_distance else None,
            'first_detection_elapsed_s': first_detection_elapsed,
            'warning_delay_s': warning_delay,
            'min_distance_m': round(min_distance, 3) if min_distance else None,
            'miss_rate': round(miss_rate, 4),
            'gt_visible_frames': gt_visible_frames,
            'false_negatives': fn_count,
            'false_positive_count': false_positive_count,
            'precision': round(precision, 4),
            'fp_per_frame': round(fp_per_frame, 4),
            'critical_zone_frames': critical_zone_frames,
            'critical_zone_misses': critical_zone_misses,
            'other_person_detections': other_person_count,
        })
        hits = {}
        for frame, type_id, actor_id in collisions:
            if frame >= start_frame and actor_id not in hits:
                hits[actor_id] = (frame - start_frame, type_id)
        summary_row['ego_collisions'] = '; '.join(f"{tid}@{n * FIXED_DELTA_S:.2f}s" for n, tid in hits.values())
        for n, tid in hits.values():
            print(f"  URTO: l'ego e' stato toccato da {tid} al tick {n} ({n * FIXED_DELTA_S:.2f} s)")

        print(f"  Risultato {run_label}: detection_rate (recall)={detection_rate:.2%} "
              f"({detection_count}/{gt_visible_frames} frame con pedone visibile), precision={precision:.2%}")
        print(f"     Sample salvati: {samples_saved}")
        print(f"     Classi rilevate (conf>={CONF_THRESHOLD}): {detection_classes}")
        print(f"     Warning delay: {warning_delay}s")
        print(f"     Miss rate (zona critica): {miss_rate:.2%} ({critical_zone_misses}/{critical_zone_frames})")
        print(f"     Falsi positivi 'person': {false_positive_count} ({fp_per_frame:.3f} per frame)")
        print(f"     Detection di altre persone reali (marciapiede, motociclista; non errori): {other_person_count}")

    except Exception as exc:
        summary_row['status'] = 'failed'
        print(f"  ERRORE {run_label}: {type(exc).__name__}: {exc}")

    finally:
        # Ripristina impostazioni originali di CARLA
        try:
            world.apply_settings(original_settings)
        except:
            pass
            
        if camera is not None:
            try:
                camera.stop()
                camera.destroy()
            except:
                pass
        if scenario is not None:
            scenario.cleanup()
        if device != 'cpu' and hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

    return summary_row, frame_rows


# ============================================================================
# -- Cella 25 del notebook ------------------------------------------------------
# ============================================================================

from collections import deque


class ForwardCollisionWarning:
    """
    Modulo di warning basato su detection confermate e time-to-collision (TTC).
    Usato come FCW per il pedone (Scenario 1) e come red light warning per il semaforo (Scenario 2).

    Ad ogni frame riceve:
        positive  : target rilevato correttamente (FCW: TP pedone; RLW: TP semaforo + stato RED)
        false_cue : oggetto inesistente che farebbe scattare il warning
                    (FCW: falso positivo nel percorso dell'ego; RLW: falso positivo classificato rosso)
        ttc_s     : time-to-collision reale = gap / velocita' (None se l'ego e' fermo)
        gap_m     : distanza reale dal punto di conflitto (paraurti-pedone / linea d'arresto)
        speed_mps : velocita' dell'ego
        elapsed_s : tempo simulato

    Logica:
        - target confermato se positive in almeno K degli ultimi N frame (filtra detection isolate)
        - zona di warning: TTC reale <= soglia
        - warning corretto: target confermato mentre si e' in zona di warning; una volta emesso
          il warning resta attivo (come in un FCW reale, anche se il target esce dall'inquadratura)
        - warning mancato: frame in zona di warning prima che il warning sia stato emesso
        - falso allarme: false_cue confermato (K su N)
    """

    def __init__(self, ttc_threshold_s=None, confirm_k=None, confirm_n=None,
                 reaction_s=None, decel_mps2=None):
        self.ttc_threshold_s = FCW_TTC_WARNING_S if ttc_threshold_s is None else ttc_threshold_s
        self.confirm_k = FCW_CONFIRM_K if confirm_k is None else confirm_k
        self.confirm_n = FCW_CONFIRM_N if confirm_n is None else confirm_n
        self.reaction_s = FCW_REACTION_S if reaction_s is None else reaction_s
        self.decel_mps2 = FCW_DECEL_MPS2 if decel_mps2 is None else decel_mps2

        self._positive_hist = deque(maxlen=self.confirm_n)
        self._false_hist = deque(maxlen=self.confirm_n)
        self._prev_false_alarm = False

        self.warning_zone_frames = 0
        self.true_positive_warnings = 0   # frame in zona con warning attivo
        self.missed_warnings = 0          # frame in zona senza warning
        self.false_alarm_frames = 0
        self.false_alarm_events = 0       # numero di allarmi falsi distinti (fronti di salita)
        self.zone_entry_time = None
        self.first_warning = None         # dict con tempo, gap, TTC, velocita' del primo warning
        self.warning_log = []

    def evaluate_frame(self, positive, false_cue, ttc_s, gap_m, speed_mps, elapsed_s):
        """Valuta un frame e restituisce lo stato del warning."""
        self._positive_hist.append(bool(positive))
        self._false_hist.append(bool(false_cue))
        confirmed = sum(self._positive_hist) >= self.confirm_k
        false_alarm = sum(self._false_hist) >= self.confirm_k

        in_zone = ttc_s is not None and not pd.isna(ttc_s) and ttc_s <= self.ttc_threshold_s
        warning_type = 'none'
        if in_zone:
            self.warning_zone_frames += 1
            if self.zone_entry_time is None:
                self.zone_entry_time = elapsed_s
            if confirmed or self.first_warning is not None:
                self.true_positive_warnings += 1
                warning_type = 'WARNING'
                if self.first_warning is None:
                    self.first_warning = {'time': elapsed_s, 'gap_m': gap_m, 'ttc_s': ttc_s,
                                          'speed_mps': speed_mps}
            else:
                self.missed_warnings += 1
                warning_type = 'MISSED'

        if false_alarm:
            self.false_alarm_frames += 1
            if not self._prev_false_alarm:
                self.false_alarm_events += 1
            if warning_type == 'none':
                warning_type = 'FALSE_ALARM'
        self._prev_false_alarm = false_alarm

        frame_result = {'warning_type': warning_type, 'in_warning_zone': in_zone,
                        'confirmed': confirmed, 'false_alarm': false_alarm}
        self.warning_log.append(frame_result)
        return frame_result

    def stopping_distance(self, speed_mps):
        """Spazio necessario per fermarsi: reazione + frenata a decelerazione costante."""
        return speed_mps * self.reaction_s + speed_mps ** 2 / (2.0 * self.decel_mps2)

    def get_summary(self, prefix='fcw_'):
        """Metriche aggregate (con prefisso: 'fcw_' per il pedone, 'rlw_' per il semaforo)."""
        tp, missed, fa = self.true_positive_warnings, self.missed_warnings, self.false_alarm_frames
        fw = self.first_warning
        delay = stop_dist = can_stop = None
        if fw is not None:
            delay = max(0.0, fw['time'] - self.zone_entry_time)
            stop_dist = self.stopping_distance(fw['speed_mps'])
            can_stop = bool(fw['gap_m'] >= stop_dist)
        elif self.warning_zone_frames > 0:
            can_stop = False  # zona di warning raggiunta senza alcun warning

        summary = {
            'warnings_issued': tp,
            'true_positives': tp,
            'missed_warnings': missed,
            'false_alarms': fa,
            'false_alarm_events': self.false_alarm_events,
            'precision': round(tp / (tp + fa), 4) if (tp + fa) else None,
            'recall': round(tp / (tp + missed), 4) if (tp + missed) else None,
            'first_warning_distance_m': round(fw['gap_m'], 2) if fw else None,
            'ttc_at_warning_s': round(fw['ttc_s'], 3) if fw else None,
            'warning_delay_s': round(delay, 3) if delay is not None else None,
            'stopping_distance_m': round(stop_dist, 2) if stop_dist is not None else None,
            'can_stop': can_stop,
            'warning_zone_frames': self.warning_zone_frames,
        }
        return {prefix + k: v for k, v in summary.items()}


def evaluate_warning(frame_rows, positive_key, false_key, prefix):
    """Applica il modulo di warning ai frame di un run e restituisce le metriche con prefisso."""
    module = ForwardCollisionWarning()
    for fr in frame_rows:
        module.evaluate_frame(positive=fr[positive_key], false_cue=fr[false_key], ttc_s=fr['ttc_s'],
                              gap_m=fr['gap_m'], speed_mps=fr['speed_mps'], elapsed_s=fr['elapsed_s'])
    return module.get_summary(prefix)


# ============================================================================
# -- Cella 27 del notebook ------------------------------------------------------
# ============================================================================

def find_lane_traffic_light(world, start_transform, max_distance_m=150.0, step_m=1.0):
    """
    Semaforo che regola la corsia dell'ego e posizione della sua linea d'arresto.
    Segue la corsia dallo spawn a passi di step_m finche' incontra uno stop waypoint
    (get_stop_waypoints) nella stessa direzione di marcia, prima dell'incrocio.

    Returns:
        (semaforo, location della linea d'arresto, distanza lungo la corsia in m) o (None, None, None)
    """
    stops = [(tl, wp) for tl in world.get_actors().filter('traffic.traffic_light')
             for wp in tl.get_stop_waypoints()]
    wp = world.get_map().get_waypoint(start_transform.location)
    travelled = 0.0
    while travelled <= max_distance_m and not wp.is_junction:
        fwd = wp.transform.get_forward_vector()
        for tl, stop_wp in stops:
            stop_fwd = stop_wp.transform.get_forward_vector()
            same_direction = fwd.x * stop_fwd.x + fwd.y * stop_fwd.y > 0.7
            if same_direction and stop_wp.transform.location.distance(wp.transform.location) <= 1.5:
                return tl, stop_wp.transform.location, travelled
        next_wps = wp.next(step_m)
        if not next_wps:
            break
        wp = next_wps[0]
        travelled += step_m
    return None, None, None


def spawn_behind(world, spawn, back_m, step_m=5.0):
    """
    Transform back_m metri PRIMA dello spawn, sulla stessa retta di marcia. Serve a dare
    all'ego lo spazio per finire la rampa di partenza prima della zona di warning.
    Ogni step_m controlla che la strada dietro sia dritta e fuori dagli incroci; se non lo e',
    restituisce lo spawn originale.
    """
    fwd = spawn.get_forward_vector()
    carla_map = world.get_map()
    d = step_m
    while d <= back_m + 1e-6:
        loc = spawn.location - fwd * d
        wp = carla_map.get_waypoint(loc)
        yaw_diff = abs((wp.transform.rotation.yaw - spawn.rotation.yaw + 180.0) % 360.0 - 180.0)
        offset = ((wp.transform.location.x - loc.x) ** 2 + (wp.transform.location.y - loc.y) ** 2) ** 0.5
        if wp.is_junction or yaw_diff > 3.0 or offset > 1.0:
            print(f"  AVVISO: strada non dritta {d:.0f} m dietro lo spawn: uso lo spawn originale")
            return spawn
        d += step_m
    return carla.Transform(spawn.location - fwd * back_m, spawn.rotation)


class TrafficLightScenario:
    """
    Scenario controllato: ego vehicle si avvicina a un semaforo rosso.
    Spawn point 103 (Town10HD_Opt). Il semaforo target e' quello che regola la corsia
    dell'ego (trovato con gli stop waypoint di CARLA), non semplicemente il piu' vicino:
    il palo dall'altra parte dell'incrocio regola la strada trasversale.
    L'ego parte TL_SPAWN_BACK_M metri prima dello spawn, per finire la rampa prima della zona di warning.

    Traffico e incidente (CROSS_TRAFFIC): l'incrocio e' a T, dopo la linea d'arresto c'e' la strada
    trasversale, con il verde. Un'auto la attraversa davanti all'ego quando e' lontano; durante
    l'avvicinamento non ci sono auto davanti (la camera deve riconoscere il semaforo da sola), due
    auto veloci arrivano dai lati fuori dall'inquadratura. L'ego passa col rosso e continua: una
    passa davanti (mancato incidente), l'altra lo urta sul lato conducente (_plan_cross_traffic).
    """
    SPAWN_INDEX = 103

    def __init__(self, world):
        self.world = world
        self.vehicles = []
        self.sensors = []
        self.target_tl = None
        self.stop_location = None   # linea d'arresto del semaforo target sulla corsia dell'ego
        self.cross_traffic = []     # veicoli sulla strada trasversale (CROSS_TRAFFIC)
        self.crash_car = None
        self.impact_frame = None    # tick previsto dell'urto
        self.collision = None
        self.blueprint_lib = world.get_blueprint_library()

    def setup(self):
        spawn_points = self.world.get_map().get_spawn_points()
        ego_spawn = spawn_behind(self.world, spawn_points[self.SPAWN_INDEX], TL_SPAWN_BACK_M)

        # Spawn ego vehicle
        vehicle_bp = self.blueprint_lib.find('vehicle.tesla.model3')
        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, ego_spawn)
        self.vehicles.append(self.ego_vehicle)

        tesla_lights = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam
        self.ego_vehicle.set_light_state(carla.VehicleLightState(tesla_lights))
        
        # Semaforo target: quello che regola la corsia dell'ego
        best_tl, self.stop_location, route_m = find_lane_traffic_light(self.world, ego_spawn)
        if best_tl:
            self.target_tl = best_tl
            # Blocca il semaforo su ROSSO
            self.target_tl.set_state(carla.TrafficLightState.Red)
            self.target_tl.set_green_time(999)
            self.target_tl.set_red_time(999)
            if CROSS_TRAFFIC:
                # Verde per la strada trasversale (prima del freeze, che blocca tutti i semafori)
                for tl in best_tl.get_group_traffic_lights():
                    if tl.id != best_tl.id:
                        tl.set_state(carla.TrafficLightState.Green)
            self.target_tl.freeze(True)
            print(f"  Semaforo target (regola la corsia dell'ego): ID={best_tl.id}, "
                  f"linea d'arresto a {route_m:.0f} m dallo spawn")
            if CROSS_TRAFFIC:
                self._plan_cross_traffic(ego_spawn)
        else:
            print("  AVVISO: nessun semaforo trovato lungo la corsia dello spawn 103.")

        return self.ego_vehicle, self.target_tl

    def _plan_cross_traffic(self, ego_spawn):
        """
        Traffico sulla strada trasversale (incrocio a T, verde per la trasversale) e incidente:
          1. auto da destra a velocita' di crociera, gia' in movimento: attraversa davanti all'ego quando
             il suo paraurti e' a CROSS_EARLY_GAP_M dalla linea d'arresto (ego ancora lontano)
          2. auto veloce da destra: parte fuori dall'inquadratura e passa davanti all'ego
             CROSS_NEAR_MISS_S dopo che ha superato la linea d'arresto (mancato incidente)
          3. auto veloce da sinistra: centra il lato conducente dell'ego dentro l'incrocio. La strada
             di sinistra e' in curva: l'auto segue i punti della corsia in modo cinematico e passa
             alla fisica vera poco prima dell'incrocio, dove la strada e' dritta
        I tempi sono calcolati dalla rampa dell'ego (time_to_travel): identici in ogni condizione.
        """
        carla_map = self.world.get_map()
        fwd, right = ego_spawn.get_forward_vector(), ego_spawn.get_right_vector()

        def rel(location):
            d = location - ego_spawn.location
            return d.x * fwd.x + d.y * fwd.y, d.x * right.x + d.y * right.y

        wp = carla_map.get_waypoint(self.stop_location)
        for _ in range(30):
            if wp.is_junction:
                break
            wp = wp.next(1.0)[0]
        if not wp.is_junction:
            print("  AVVISO: incrocio dopo la linea d'arresto non trovato, niente traffico trasversale")
            return

        # Corsie che attraversano l'incrocio dritte e perpendicolari alla marcia dell'ego:
        # +90 = da sinistra verso destra, -90 = da destra verso sinistra (ordinate dalla piu' vicina all'ego)
        lanes = {90: [], -90: []}
        for entry, exit_wp in wp.get_junction().get_waypoints(carla.LaneType.Driving):
            yaw_in = (entry.transform.rotation.yaw - ego_spawn.rotation.yaw + 180.0) % 360.0 - 180.0
            yaw_out = (exit_wp.transform.rotation.yaw - ego_spawn.rotation.yaw + 180.0) % 360.0 - 180.0
            if abs((yaw_out - yaw_in + 180.0) % 360.0 - 180.0) > 15.0:
                continue   # svolta
            for side in lanes:
                if abs(yaw_in - side) < 15.0:
                    lanes[side].append(entry)
        left_lanes = sorted(lanes[90], key=lambda e: rel(e.transform.location)[0])
        right_lanes = sorted(lanes[-90], key=lambda e: rel(e.transform.location)[0])
        if not left_lanes or len(right_lanes) < 2:
            print("  AVVISO: corsie trasversali non trovate, niente traffico trasversale")
            return

        ego_bb = self.ego_vehicle.bounding_box.extent
        stop_fwd = rel(self.stop_location)[0]
        t_stop = time_to_travel(stop_fwd - ego_bb.x)
        fast = CROSS_FAST_KMH / 3.6

        # 1. Da destra, gia' in movimento (corsia esterna)
        entry = right_lanes[-1]
        t_cross = time_to_travel(stop_fwd - ego_bb.x - CROSS_EARLY_GAP_M)
        back = EGO_SPEED_MPS * t_cross - abs(rel(entry.transform.location)[1])
        self._add_cross_car('vehicle.lincoln.mkz_2020', entry.previous(back)[0].transform, EGO_SPEED_MPS, 1,
                            f"da destra, gia' in movimento: passa davanti all'ego con il paraurti "
                            f"a {CROSS_EARLY_GAP_M:.0f} m dalla linea d'arresto")

        # 2. Da destra, veloce: mancato incidente (corsia interna)
        entry = right_lanes[0]
        dist = CROSS_FAST_PARK_M + abs(rel(entry.transform.location)[1])
        launch = int(round((t_stop + CROSS_NEAR_MISS_S - dist / fast) / FIXED_DELTA_S))
        self._add_cross_car('vehicle.audi.etron', entry.previous(CROSS_FAST_PARK_M)[0].transform, fast, launch,
                            f"da destra a {CROSS_FAST_KMH:.0f} km/h: passa davanti all'ego "
                            f"{CROSS_NEAR_MISS_S:.1f} s dopo la linea d'arresto")

        # 3. Da sinistra, veloce: urto sul lato conducente (corsia piu' vicina all'ego)
        entry = left_lanes[0]
        step = 0.5
        back_pts = [entry.previous(step * k)[0] for k in range(int(CROSS_KINEMATIC_BACK_M / step), 0, -1)]
        fwd_pts = [entry] + [entry.next(step * k)[0] for k in range(1, int(30.0 / step))]
        path = [p.transform for p in back_pts + fwd_pts]
        s = [0.0]
        for a, b in zip(path, path[1:]):
            s.append(s[-1] + a.location.distance(b.location))
        s_entry = s[len(back_pts)]
        i_axis = next(i for i, tf in enumerate(path) if rel(tf.location)[1] >= 0.0)
        car = self._add_cross_car('vehicle.dodge.charger_2020', path[0], fast, None,
                                  f"da sinistra a {CROSS_FAST_KMH:.0f} km/h: urta il lato conducente dell'ego")
        if car is None:
            return
        car_bb = car['actor'].bounding_box.extent
        t_imp = time_to_travel(rel(path[i_axis].location)[0])   # il centro dell'ego arriva sulla corsia dell'auto
        car.update(path=path, s=s, s_imp=s[i_axis] - ego_bb.y - car_bb.x, t_imp=t_imp,
                   s_switch=s_entry - 2.0, mode='kinematic')
        car['actor'].set_simulate_physics(False)
        self.crash_car = car
        self.impact_frame = int(round(t_imp / FIXED_DELTA_S))

    def _add_cross_car(self, name, transform, speed_mps, launch_frame, label):
        """Spawna un veicolo fermo (frenato) che partira' a speed_mps al tick launch_frame."""
        transform = carla.Transform(transform.location + carla.Location(z=0.5), transform.rotation)
        blueprints = self.blueprint_lib.filter(name) or self.blueprint_lib.filter('vehicle.audi.etron')
        vehicle = self.world.try_spawn_actor(blueprints[0], transform)
        if vehicle is None:
            print(f"  AVVISO: spawn del veicolo trasversale fallito ({label})")
            return None
        vehicle.set_light_state(carla.VehicleLightState(
            carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam))
        vehicle.apply_control(carla.VehicleControl(brake=1.0))
        self.vehicles.append(vehicle)
        car = {'actor': vehicle, 'speed': speed_mps, 'launch_frame': launch_frame, 'mode': 'parked'}
        self.cross_traffic.append(car)
        print(f"  Traffico: {vehicle.type_id} {label}")
        return car

    @staticmethod
    def _path_transform(car, s_now):
        """Transform lungo il percorso cinematico alla distanza s_now (interpolazione lineare)."""
        path, s = car['path'], car['s']
        i = max(0, min(len(s) - 2, next((k for k in range(len(s) - 1) if s[k + 1] >= s_now), len(s) - 2)))
        a, b = path[i].location, path[i + 1].location
        f = min(1.0, max(0.0, (s_now - s[i]) / max(s[i + 1] - s[i], 1e-6)))
        loc = carla.Location(a.x + (b.x - a.x) * f, a.y + (b.y - a.y) * f, a.z + (b.z - a.z) * f + 0.05)
        yaw = np.degrees(np.arctan2(b.y - a.y, b.x - a.x))
        return carla.Transform(loc, carla.Rotation(yaw=float(yaw)))

    def update_traffic(self, n):
        """
        Da chiamare prima del tick n (n = 1, 2, ... dalla partenza dell'ego): fa partire i veicoli
        trasversali, muove quello cinematico e, all'urto, lascia la dinamica alla fisica di CARLA.
        """
        for car in self.cross_traffic:
            actor = car['actor']
            if car['mode'] == 'parked' and car['launch_frame'] is not None and n >= car['launch_frame']:
                actor.apply_control(carla.VehicleControl(brake=0.0))
                actor.enable_constant_velocity(carla.Vector3D(car['speed'], 0.0, 0.0))
                car['mode'] = 'constant'
            elif car['mode'] == 'kinematic':
                s_now = car['s_imp'] - car['speed'] * (car['t_imp'] - n * FIXED_DELTA_S)
                actor.set_transform(self._path_transform(car, max(0.0, s_now)))
                if s_now >= car['s_switch']:   # strada dritta: da qui fisica vera
                    actor.set_simulate_physics(True)
                    actor.enable_constant_velocity(carla.Vector3D(car['speed'], 0.0, 0.0))
                    car['mode'] = 'constant'

        # Urto (contatto tra il muso dell'auto da sinistra e il fianco sinistro dell'ego): da qui
        # niente piu' velocita' imposte, la dinamica dell'impatto e' quella fisica di CARLA
        if self.crash_car is not None and self.collision is None:
            ego_tf = self.ego_vehicle.get_transform()
            d = self.crash_car['actor'].get_location() - ego_tf.location
            f, r = ego_tf.get_forward_vector(), ego_tf.get_right_vector()
            lon, lat = d.x * f.x + d.y * f.y, d.x * r.x + d.y * r.y
            ego_bb, car_bb = self.ego_vehicle.bounding_box.extent, self.crash_car['actor'].bounding_box.extent
            if abs(lat) <= ego_bb.y + car_bb.x + 0.3 and abs(lon) <= ego_bb.x + car_bb.y:
                self.collision = {'frame': n, 'with': self.crash_car['actor'].type_id,
                                  'point_m': round(lon, 2)}   # >0: davanti al centro dell'ego
                for vehicle in self.vehicles:
                    vehicle.disable_constant_velocity()
                print(f"  URTO al tick {n}: {self.collision['with']} sul fianco sinistro dell'ego "
                      f"({lon:+.1f} m dal centro)")

    def post_stop_frames(self, frame_count):
        """Tick da simulare dopo la linea d'arresto: fino a TL_POST_CRASH_S dopo l'urto previsto."""
        if self.impact_frame is None:
            return 0
        return max(0, self.impact_frame + int(round(TL_POST_CRASH_S / FIXED_DELTA_S)) - frame_count)

    def set_weather(self, weather_type):
        """Applica condizione meteo (backlight orientato sullo yaw dell'ego) e restituisce i parametri."""
        self.weather_params = weather_params(weather_type, ego_yaw=self.ego_vehicle.get_transform().rotation.yaw,
                                             spawn_index=self.SPAWN_INDEX)
        self.world.set_weather(carla.WeatherParameters(**self.weather_params))
        self.weather_params['street_lights'] = set_street_lights(self.world, weather_type, self.weather_params)
        self.weather_params['ego_lights'] = set_ego_lights(self.ego_vehicle, weather_type)
        return self.weather_params

    def start_motion(self):
        """Rilascia i freni dell'ego dopo l'assestamento: da qui la velocita' e' imposta con set_speed()."""
        self._speed = None
        self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0))

    def set_speed(self, speed_mps):
        """
        Velocita' imposta all'ego (enable_constant_velocity, coordinate locali); il traffico
        trasversale ha velocita' proprie (update_traffic). Chiamata a ogni frame con la rampa di
        ramp_speed(): la chiamata al server parte solo se il valore cambia. Dopo l'urto non
        reimpone piu' nulla.
        """
        if speed_mps == self._speed or self.collision is not None:
            return
        self.ego_vehicle.enable_constant_velocity(carla.Vector3D(speed_mps, 0.0, 0.0))
        self._speed = speed_mps

    def cleanup(self):
        for actor in self.sensors + self.vehicles:
            if actor.is_alive:
                actor.destroy()
        if self.target_tl:
            try:
                self.target_tl.freeze(False)
            except:
                pass
        try:
            settings = self.world.get_settings()
            was_sync = settings.synchronous_mode
            settings.synchronous_mode = True
            self.world.apply_settings(settings)
            self.world.tick()
            settings.synchronous_mode = was_sync
            self.world.apply_settings(settings)
        except:
            pass


# ============================================================================
# -- Cella 28 del notebook ------------------------------------------------------
# ============================================================================

def classify_traffic_light_state(tl_crop_rgb):
    import cv2
    import numpy as np
    if tl_crop_rgb.size == 0:
        return "UNKNOWN", 0.0
    h, w, _ = tl_crop_rgb.shape
    if h < 6 or w < 2:
        return "UNKNOWN", 0.0
    h_third = h // 3
    zones = {
        'RED': tl_crop_rgb[0:h_third, :, :],
        'YELLOW': tl_crop_rgb[h_third:2*h_third, :, :],
        'GREEN': tl_crop_rgb[2*h_third:h, :, :]
    }
    scores = {}
    for state, zone in zones.items():
        if zone.size == 0:
            scores[state] = 0
            continue
        hsv = cv2.cvtColor(zone, cv2.COLOR_RGB2HSV)
        v = hsv[:,:,2]
        s = hsv[:,:,1]
        # Active lamp criteria
        mask = (v > 120) & (s > 50)
        scores[state] = np.sum(v[mask])
    
    best_state = max(scores, key=scores.get)
    best_score = scores[best_state]
    if best_score == 0:
        return "UNKNOWN", 0.0
    total = sum(scores.values())
    conf = best_score / total if total > 0 else 0.0
    return best_state, float(conf)

def draw_traffic_light_box(image_rgb, xyxy, state, conf):
    import cv2
    color_map = {
        'RED': (255, 0, 0),
        'YELLOW': (255, 255, 0),
        'GREEN': (0, 255, 0),
        'UNKNOWN': (128, 128, 128)
    }
    c = color_map.get(state, (128, 128, 128))
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    cv2.rectangle(image_rgb, (x1, y1), (x2, y2), c, 2)
    label = f"{state} {conf:.2f}"
    cv2.putText(image_rgb, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    return image_rgb

def get_custom_annotated(result, tl_state_data, gt_boxes=()):
    if result is None:   # senza camera
        return None
    img = annotate_frame(result, gt_boxes).copy()
    for (xyxy, state, conf) in tl_state_data:
        img = draw_traffic_light_box(img, xyxy, state, conf)
    return img

def traffic_light_heads(world, center, radius_m=150.0):
    """
    Box 3D (vertici in coordinate mondo) di ogni lanterna dei semafori entro radius_m.
    I semafori sono statici: si calcolano una sola volta per run.

    Returns:
        dict {(tl_id, indice_lanterna): array (8, 3)}
    """
    heads = {}
    for tl in world.get_actors().filter('traffic.traffic_light'):
        if tl.get_location().distance(center) > radius_m:
            continue
        # get_light_boxes() restituisce box gia' in coordinate mondo -> trasformazione identita'
        for k, bb in enumerate(tl.get_light_boxes()):
            heads[(tl.id, k)] = vertices_to_array(bb.get_world_vertices(carla.Transform()))
    return heads

def run_condition_tl(world, model, batch_id, weather, corruption, severity, samples_dir, camera_enabled=True):
    """
    Esegue un test completo per lo Scenario 2: Traffic Light Detection.
    Connessione camera e spectator chase-cam sincrone identiche allo Scenario 1.
    Target YOLO: classe COCO 'traffic light' (id 9).

    Ground truth: le lanterne di TUTTI i semafori vicini vengono proiettate nell'immagine.
    - TP: una lanterna del semaforo target e' visibile ed e' associata a una detection
    - FN: lanterna target visibile ma non rilevata
    - FP: detection 'traffic light' che non corrisponde a nessuna lanterna reale
    - le detection di ALTRI semafori reali non sono errori: contate a parte (other_tl_detections)
    Lo stato (colore) viene classificato solo sulla detection associata al target.

    Distanze: dal paraurti anteriore alla linea d'arresto del semaforo target, lungo la direzione di marcia.
    Il run termina quando l'ego arriva sulla linea d'arresto (non entra nell'incrocio).
    Immagini salvate a: partenza, 30 m, 20 m, 10 m, linea d'arresto.
    """
    run_label = f"tl_{weather}_{corruption}{severity}"
    print(f"\n{'='*50}")
    print(f"  Running: {run_label}")

    # Distanze dalla linea d'arresto (paraurti anteriore)
    CRITICAL_DISTANCE_M = round(FCW_TTC_WARNING_S * EGO_SPEED_MPS, 1)  # zona critica = zona di warning (2.5 s)
    WARNING_DISTANCE_M = CRITICAL_DISTANCE_M + 10.0                    # inizio avvicinamento
    STOP_MARGIN_M = 0.5         # fine del run: ego sulla linea d'arresto

    frame_rows = []
    summary_row = {
        'batch_id': batch_id,
        'run_label': run_label,
        'weather': weather,
        'corruption': corruption,
        'severity': severity,
        'total_frames': 0,
        'detection_frames': 0,
        'detection_rate': 0.0,
        'state_accuracy': 0.0,
        'first_detection_distance_m': None,
        'first_detection_elapsed_s': None,
        'warning_delay_s': None,
        'min_distance_m': None,
        'miss_rate': 0.0,
        'gt_visible_frames': 0,
        'false_negatives': 0,
        'false_positive_count': 0,
        'precision': 0.0,
        'fp_per_frame': 0.0,
        'other_tl_detections': 0,
        'critical_zone_frames': 0,
        'critical_zone_misses': 0,
        'target_tl_id': None,
        'lane_tl_id': None,          # semaforo che CARLA associa all'ego sulla linea d'arresto (verifica)
        'start_distance_m': None,    # distanza iniziale dalla linea d'arresto
        'weather_params': None,
        'weather_ok': None,
        'status': 'ok',
    }

    scenario = None
    camera = None
    first_detection_distance = None
    first_detection_time = None
    entered_warning_time = None
    frame_count = 0
    detection_count = 0      # frame TP
    fn_count = 0             # frame FN
    gt_visible_frames = 0    # frame con il semaforo target visibile
    min_distance = None
    critical_zone_frames = 0
    critical_zone_misses = 0
    false_positive_count = 0
    other_tl_count = 0
    state_correct_count = 0
    state_evaluated_count = 0
    detection_classes = {}

    # Sotto-cartella per sample di questa condizione
    condition_samples_dir = os.path.join(samples_dir, run_label)
    os.makedirs(condition_samples_dir, exist_ok=True)
    samples_saved = 0

    # Definiti prima del try: il blocco finally li usa anche se il setup fallisce
    original_settings = world.get_settings()
    device = select_device()
    use_half = YOLO_HALF and device.startswith('cuda')

    try:
        # Abilita Synchronous Mode (come in Scenario 1)
        enable_sync_mode(world)
        settings = world.get_settings()
        print(f"  Modalita' sincrona: {settings.synchronous_mode}, passo fisso {settings.fixed_delta_seconds} s")

        spectator = world.get_spectator()

        # Setup scenario
        scenario = TrafficLightScenario(world)
        ego_vehicle, target_tl = scenario.setup()
        summary_row['weather_params'] = json.dumps(scenario.set_weather(weather))
        if target_tl is None:
            raise RuntimeError("semaforo target non trovato: impossibile calcolare la ground truth")

        # Attach camera RGB (senza camera: niente immagini ne' YOLO, solo moto e ground truth)
        image_queue = None
        if camera_enabled:
            camera = spawn_rgb_camera(world, scenario.ego_vehicle)

            # Coda: ogni immagine viene associata al tick che l'ha generata
            image_queue = queue.Queue()
            camera.listen(image_queue.put)
        scenario.ego_vehicle.set_autopilot(False)

        # Settling phase
        for _ in range(10):
            scenario.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
            tick_and_get_frame(world, image_queue, scenario.ego_vehicle)
            move_spectator_to(scenario.ego_vehicle.get_transform(), spectator, distance=8.0, z=3.0, pitch=-20)

        summary_row['weather_ok'] = check_applied_weather(world, scenario.weather_params, weather)

        # Partenza: ego a velocita' costante
        scenario.start_motion()
        drift = DriftTracker(scenario.ego_vehicle.get_transform())
        sim_start_time = world.get_snapshot().timestamp.elapsed_seconds

        # Ground truth: parametri camera e lanterne dei semafori vicini
        K = camera_intrinsics(WIDTH, HEIGHT, FOV)
        tl_heads = traffic_light_heads(world, scenario.ego_vehicle.get_location())
        target_id = target_tl.id
        summary_row['target_tl_id'] = target_id
        stop_loc = scenario.stop_location
        fwd0 = scenario.ego_vehicle.get_transform().get_forward_vector()
        ego_half_length = scenario.ego_vehicle.bounding_box.extent.x
        n_target_heads = sum(1 for h in tl_heads if h[0] == target_id)
        print(f"  Ground truth: {len(tl_heads)} lanterne di semafori vicini, {n_target_heads} del semaforo target")

        # Stato reale del semaforo target (bloccato su rosso nel setup)
        gt_state_str = {
            carla.TrafficLightState.Red: "RED",
            carla.TrafficLightState.Yellow: "YELLOW",
            carla.TrafficLightState.Green: "GREEN",
        }.get(target_tl.get_state(), "UNKNOWN")

        key_samples = {}

        # Main loop simulazione
        while frame_count < MAX_FRAMES_PER_RUN:
            # Velocita': rampa dolce da fermo, poi costante (stessa in ogni condizione)
            scenario.set_speed(ramp_speed(frame_count + 1))
            scenario.update_traffic(frame_count + 1)
            # Tick + immagine dello STESSO tick
            current_frame, snapshot, cam_tf = tick_and_get_frame(world, image_queue, scenario.ego_vehicle)

            # Segui il veicolo con la camera spectator
            move_spectator_to(scenario.ego_vehicle.get_transform(), spectator, distance=8.0, z=3.0, pitch=-20)

            frame_count += 1
            processed_frame = apply_corruption(current_frame, corruption, severity) if current_frame is not None else None

            # Distanza paraurti anteriore -> linea d'arresto, lungo la direzione di marcia
            v_loc = scenario.ego_vehicle.get_location()
            distance = ((stop_loc.x - v_loc.x) * fwd0.x + (stop_loc.y - v_loc.y) * fwd0.y) - ego_half_length
            if frame_count == 1:
                summary_row['start_distance_m'] = round(distance, 2)
                print(f"  Linea d'arresto a {distance:.1f} m dal paraurti")

            min_distance = distance if min_distance is None else min(min_distance, distance)

            # Velocita' e time-to-collision rispetto alla linea d'arresto
            vel = scenario.ego_vehicle.get_velocity()
            speed = float(np.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2))
            lateral_offset, heading_change = drift(scenario.ego_vehicle.get_transform())
            gap = max(0.0, distance)
            ttc = gap / speed if speed > 0.1 else None
            elapsed_s = snapshot.timestamp.elapsed_seconds - sim_start_time  # tempo SIMULATO

            # Tracking warning zone (WARNING_DISTANCE_M dalla linea d'arresto)
            if distance <= WARNING_DISTANCE_M and entered_warning_time is None:
                entered_warning_time = elapsed_s

            # Ground truth: box 2D delle lanterne visibili in questo frame
            gt_boxes = {}
            for head_id, verts in tl_heads.items():
                box = project_box_2d(verts, cam_tf, K, WIDTH, HEIGHT)
                if box is not None:
                    gt_boxes[head_id] = box
            target_gt = [box for head_id, box in gt_boxes.items() if head_id[0] == target_id]
            gt_visible = len(target_gt) > 0

            # YOLO detection (box da YOLO_MIN_CONF in su: la soglia si applica dopo)
            if processed_frame is not None:
                results = model(yolo_input(processed_frame), verbose=False, device=device,
                                conf=YOLO_MIN_CONF, imgsz=YOLO_IMGSZ, half=use_half)
                result = results[0]

                # Tutti i box in numpy con un solo trasferimento dalla GPU
                boxes = boxes_array(result)
            else:
                # Senza camera: nessuna detection
                result = None
                boxes = np.zeros((0, 6), dtype=np.float32)
            count_classes(boxes, model.names, detection_classes, CONF_THRESHOLD)

            # Associazione detection 'traffic light' <-> lanterne reali
            tl_dets = detections_of_class(boxes, model.names, 'traffic light')
            matches, unmatched = match_detections(tl_dets, gt_boxes)
            target_matches = [m for head_id, m in matches.items() if head_id[0] == target_id]
            target_conf, target_iou, target_box = max(target_matches, key=lambda m: m[0]) if target_matches else (0.0, 0.0, None)
            detected = gt_visible and target_conf >= CONF_THRESHOLD   # TP
            missed = gt_visible and not detected                       # FN
            fp_confs = [c for c, _ in unmatched]
            frame_fp = sum(c >= CONF_THRESHOLD for c in fp_confs)
            frame_other_tl = sum(1 for head_id, m in matches.items()
                                 if head_id[0] != target_id and m[0] >= CONF_THRESHOLD)
            tl_detected_count = sum(c >= CONF_THRESHOLD for c, _ in tl_dets)

            # Stato (colore) classificato solo sulla detection del semaforo target
            tl_state_data = []
            detected_state = "UNKNOWN"
            state_conf = 0.0
            if detected:
                x1, y1, x2, y2 = [int(v) for v in target_box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(processed_frame.shape[1], x2), min(processed_frame.shape[0], y2)
                detected_state, state_conf = classify_traffic_light_state(processed_frame[y1:y2, x1:x2])
                tl_state_data.append(((x1, y1, x2, y2), detected_state, state_conf))
            state_correct = int(detected_state == gt_state_str) if gt_state_str != "UNKNOWN" and detected_state != "UNKNOWN" else 0

            # Red light warning: positivo se il semaforo target e' rilevato E classificato rosso;
            # falso allarme se una detection che non e' un semaforo reale viene classificata rossa
            rlw_positive = detected and detected_state == 'RED'
            fp_red = False
            for c, box in unmatched:
                if c < CONF_THRESHOLD:
                    continue
                bx1, by1, bx2, by2 = [int(v) for v in box]
                bx1, by1 = max(0, bx1), max(0, by1)
                if classify_traffic_light_state(processed_frame[by1:by2, bx1:bx2])[0] == 'RED':
                    fp_red = True

            # Error rate tracking
            false_positive_count += frame_fp
            other_tl_count += frame_other_tl
            if gt_visible:
                gt_visible_frames += 1
            if missed:
                fn_count += 1

            # Zona critica (CRITICAL_DISTANCE_M dalla linea d'arresto): solo frame con semaforo target visibile
            in_critical_zone = distance <= CRITICAL_DISTANCE_M
            if in_critical_zone and gt_visible:
                critical_zone_frames += 1
                if missed:
                    critical_zone_misses += 1

            if detected:
                detection_count += 1
                if detected_state != "UNKNOWN" and gt_state_str != "UNKNOWN":
                    state_evaluated_count += 1
                    state_correct_count += state_correct
                if first_detection_distance is None:
                    first_detection_distance = distance
                    first_detection_time = elapsed_s
                    print(f"  Prima detection SEMAFORO: {distance:.1f}m (conf={target_conf:.2f}, IoU={target_iou:.2f})")

            # Frame logging (gt_*: box della lanterna target piu' alta nell'immagine)
            gt_main = max(target_gt, key=lambda b: b[3] - b[1]) if gt_visible else None
            frame_rows.append({
                'batch_id': batch_id,
                'run_label': run_label,
                'weather': weather,
                'corruption': corruption,
                'severity': severity,
                'frame_index': frame_count,
                'elapsed_s': round(elapsed_s, 3),
                'distance_m': round(distance, 3),
                'traffic_lights_detected': tl_detected_count,
                'max_confidence': round(target_conf, 4),  # confidenza della detection associata al target
                'detected': int(detected),
                'in_critical_zone': int(in_critical_zone),
                'gt_visible': int(gt_visible),
                'gt_heads_visible': len(target_gt),
                'gt_x1': round(gt_main[0], 1) if gt_visible else None,
                'gt_y1': round(gt_main[1], 1) if gt_visible else None,
                'gt_x2': round(gt_main[2], 1) if gt_visible else None,
                'gt_y2': round(gt_main[3], 1) if gt_visible else None,
                'gt_height_px': round(gt_main[3] - gt_main[1], 1) if gt_visible else None,
                'target_iou': round(target_iou, 3),
                'tp': int(detected),
                'fn': int(missed),
                'fp': frame_fp,
                'fp_confs': format_confs(fp_confs),
                'other_tl_detections': frame_other_tl,
                'detected_state': detected_state,
                'gt_state': gt_state_str,
                'state_confidence': round(state_conf, 4),
                'state_correct': state_correct,
                'rlw_positive': int(rlw_positive),
                'fp_red': int(fp_red),
                'speed_mps': round(speed, 3),
                'lateral_offset_m': round(lateral_offset, 3),
                'heading_change_deg': round(heading_change, 3),
                'gap_m': round(gap, 3),
                'ttc_s': round(ttc, 3) if ttc is not None else None,
            })

            # Cattura campioni visivi di avvicinamento a intervalli di distanza regolari (prima del semaforo)
            # Box verdi = ground truth delle lanterne del semaforo target
            if frame_count == 1 and '01_start' not in key_samples:
                key_samples['01_start'] = (get_custom_annotated(result, tl_state_data, target_gt), f"01_start_{distance:.1f}m")
            elif distance <= 30.0 and '02_approach' not in key_samples:
                key_samples['02_approach'] = (get_custom_annotated(result, tl_state_data, target_gt), f"02_approach_{distance:.1f}m")
            elif distance <= 20.0 and '03_approach' not in key_samples:
                key_samples['03_approach'] = (get_custom_annotated(result, tl_state_data, target_gt), f"03_approach_{distance:.1f}m")
            elif distance <= 10.0 and '04_near' not in key_samples:
                key_samples['04_near'] = (get_custom_annotated(result, tl_state_data, target_gt), f"04_near_{distance:.1f}m")

            # Traccia eventuale prima detection e miss in zona critica
            if detected and 'detected' not in key_samples:
                key_samples['detected'] = (get_custom_annotated(result, tl_state_data, target_gt), f"det_first_{distance:.1f}m_conf{target_conf:.2f}")
            elif in_critical_zone and missed and 'miss' not in key_samples:
                key_samples['miss'] = (get_custom_annotated(result, tl_state_data, target_gt), f"miss_crit_{distance:.1f}m")

            # Fine delle metriche: ego sulla linea d'arresto
            if distance <= STOP_MARGIN_M:
                key_samples['05_final_stop'] = (get_custom_annotated(result, tl_state_data, target_gt), f"05_stopline_{distance:.1f}m")
                # Verifica: il semaforo che CARLA associa all'ego deve essere il target
                lane_tl = scenario.ego_vehicle.get_traffic_light()
                summary_row['lane_tl_id'] = lane_tl.id if lane_tl is not None else None
                print(f"  Veicolo arrivato alla linea d'arresto ({distance:.1f} m) -> fine metriche")

                # Con il traffico trasversale l'ego passa col rosso e la simulazione continua fino
                # a TL_POST_CRASH_S dopo l'urto (solo visualizzazione: nessuna metrica)
                post_frames = scenario.post_stop_frames(frame_count) if CROSS_TRAFFIC else 0
                for k in range(frame_count + 1, frame_count + 1 + post_frames):
                    scenario.set_speed(ramp_speed(k))
                    scenario.update_traffic(k)
                    tick_and_get_frame(world, image_queue, scenario.ego_vehicle)
                    move_spectator_to(scenario.ego_vehicle.get_transform(), spectator, distance=8.0, z=3.0, pitch=-20)
                if scenario.collision is not None:
                    summary_row['collision_with'] = scenario.collision['with']
                    summary_row['collision_time_s'] = round(scenario.collision['frame'] * FIXED_DELTA_S, 2)
                    summary_row['collision_point_m'] = scenario.collision['point_m']
                break

        # Salva campioni su disco alla fine della corsa
        for s_type, (s_res, s_label) in key_samples.items():
            try:
                s_path = os.path.join(condition_samples_dir, f"{s_label}.png")
                save_annotated_sample(s_res, s_path)
                samples_saved += 1
            except Exception:
                pass

        # Calcolo metriche
        # detection_rate = recall sui frame in cui il semaforo target e' visibile
        detection_rate = detection_count / gt_visible_frames if gt_visible_frames else 0.0
        miss_rate = critical_zone_misses / critical_zone_frames if critical_zone_frames else 0.0
        precision = (detection_count / (detection_count + false_positive_count)
                     if (detection_count + false_positive_count) else 0.0)
        fp_per_frame = false_positive_count / frame_count if frame_count else 0.0

        warning_delay = None
        if first_detection_time is not None and entered_warning_time is not None:
            warning_delay = round(max(0.0, first_detection_time - entered_warning_time), 3)

        first_detection_elapsed = None
        if first_detection_time is not None:
            first_detection_elapsed = round(first_detection_time, 3)

        state_accuracy = state_correct_count / state_evaluated_count if state_evaluated_count > 0 else 0.0

        summary_row.update({
            'total_frames': frame_count,
            'detection_frames': detection_count,
            'state_accuracy': round(state_accuracy, 4),
            'detection_rate': round(detection_rate, 4),
            'first_detection_distance_m': round(first_detection_distance, 3) if first_detection_distance else None,
            'first_detection_elapsed_s': first_detection_elapsed,
            'warning_delay_s': warning_delay,
            'min_distance_m': round(min_distance, 3) if min_distance else None,
            'miss_rate': round(miss_rate, 4),
            'gt_visible_frames': gt_visible_frames,
            'false_negatives': fn_count,
            'false_positive_count': false_positive_count,
            'precision': round(precision, 4),
            'fp_per_frame': round(fp_per_frame, 4),
            'other_tl_detections': other_tl_count,
            'critical_zone_frames': critical_zone_frames,
            'critical_zone_misses': critical_zone_misses,
        })

        print(f"  Risultato {run_label}: detection_rate (recall)={detection_rate:.2%} "
              f"({detection_count}/{gt_visible_frames} frame con semaforo visibile), precision={precision:.2%}")
        print(f"     Sample salvati: {samples_saved}")
        print(f"     Classi rilevate (conf>={CONF_THRESHOLD}): {detection_classes}")
        if first_detection_distance:
            print(f"     Prima detection semaforo: {first_detection_distance:.1f}m")
        print(f"     Miss rate (zona critica): {miss_rate:.2%} ({critical_zone_misses}/{critical_zone_frames})")
        print(f"     Falsi positivi 'traffic light': {false_positive_count} ({fp_per_frame:.3f} per frame)")
        print(f"     Detection di altri semafori reali (non errori): {other_tl_count}")
        print(f"     Accuratezza stato (colore): {state_accuracy:.2%} su {state_evaluated_count} frame")

    except Exception as exc:
        summary_row['status'] = 'failed'
        print(f"  ERRORE {run_label}: {type(exc).__name__}: {exc}")

    finally:
        try:
            world.apply_settings(original_settings)
        except:
            pass
        if camera is not None:
            try:
                camera.stop()
                camera.destroy()
            except:
                pass
        if scenario is not None:
            scenario.cleanup()
        if device != 'cpu' and hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

    return summary_row, frame_rows
