"""Central configuration for the cyber-resilient smart-metering simulator.

Every number used to generate the dataset lives here so that the experiment
is fully specified by this file plus the random seed.
"""
from dataclasses import dataclass, field

SEED = 20261019          # master seed; all generators derive from it
DT = 5                   # reporting interval, seconds
SPD = 86400 // DT        # samples per day (17,280)
DAYS = 60                # 30-day baseline + 30-day optimisation
BASE_DAYS = 30
V_LL = 415.0             # nominal line-to-line voltage, V

@dataclass
class Machine:
    name: str
    rated_kw: float      # rated active power
    sec_base: float      # true specific energy consumption, kWh/kg (baseline)
    sec_gain: float      # SEC multiplier after optimisation (maintenance fixes)
    pf_base: float       # full-load power factor before correction
    shift_start_h: float # production shift start (hour of day)

MACHINES = [
    Machine("HM-1", 45.0, 0.531, 0.965, 0.89, 8.0),
    Machine("HM-2", 50.0, 0.596, 0.940, 0.87, 8.0),
    Machine("HM-3", 55.0, 0.530, 0.975, 0.88, 8.0),
    Machine("HM-4", 60.0, 0.635, 0.860, 0.88, 8.0),
]

@dataclass
class PlantParams:
    # --- operating schedule -------------------------------------------------
    shift_hours: float = 12.0          # 08:00-20:00
    heatup_min: float = 35.0           # heat-up after start, at ~95 % rating
    stagger_min: float = 10.0          # start offset between machines (optimised)
    load_mean: float = 0.34            # mean production load (fraction of rating)
    load_weibull_k: float = 1.5        # Weibull shape for load fluctuation
    load_weibull_scale: float = 0.18   # Weibull scale (fraction of rating)
    idle_frac: float = 0.33            # idle draw (heaters held at temperature)
    standby_frac: float = 0.06         # standby draw after operator action
    off_leak_kw: float = 0.5           # parasitic leakage when "off"
    # baseline idle behaviour
    lunch_h: float = 13.0
    lunch_min: float = 60.0
    shiftchange_h: float = 16.0
    shiftchange_min: float = 30.0
    random_idle_per_day: float = 3.0   # Poisson rate of material-wait idles
    random_idle_mean_min: float = 25.0
    overrun_prob: float = 0.6          # machine left idle after shift end
    overrun_min: tuple = (30.0, 120.0)
    # optimised behaviour: operator reacts to idle alert after this delay
    alert_response_min: float = 8.0
    response_prob: float = 0.8         # share of idle alerts acted upon
    demand_sigma: float = 0.07         # day-to-day production-demand variation
    pf_load_slope: float = 0.06
    # startup transient
    inrush_mult: tuple = (1.7, 2.0)    # peak kVA multiple of rating
    inrush_tau_s: float = 45.0
    # power factor
    pf_idle: float = 0.76
    pf_drift_per_day: float = 0.0004
    pf_corrected: float = 0.955        # after per-machine capacitor banks
    # noise / waveform
    v_sigma: float = 0.012
    imb_normal: float = 0.025          # typical current imbalance
    thd_normal: tuple = (0.045, 0.075)
    aux_load_kw: float = 18.0          # unmetered auxiliary load on feeder
    aux_sigma: float = 0.015           # feeder-level unexplained variation

@dataclass
class EventParams:
    # electrical-health episodes per machine over the full horizon
    imbalance_n: int = 4
    imbalance_min: tuple = (20, 120)
    thd_n: int = 3
    thd_min: tuple = (30, 180)
    overload_n: int = 6
    overload_min: tuple = (5, 20)
    pfdrop_n: int = 3                  # capacitor fuse failure (optimised period)
    pfdrop_min: tuple = (60, 360)
    # energy-inefficiency episodes (heater fault, power +15-30 % for same output)
    secfault_n: int = 4
    secfault_min: tuple = (60, 480)

@dataclass
class NetParams:
    latency_ms: tuple = (20.0, 250.0)  # normal end-to-end latency range
    loss_rate: float = 0.005           # benign random packet loss
    burst_per_day: float = 2.0         # EMI / Wi-Fi bursts per machine-day
    burst_s: tuple = (5.0, 60.0)
    burst_delay_s: tuple = (1.0, 14.0) # retransmission delay inside bursts
    clock_sigma_s: float = 0.25        # residual NTP offset
    freshness_s: float = 10.0          # Algorithm 2 freshness window
    heartbeat_tol_s: float = 30.0      # gap that raises an availability alert

@dataclass
class AttackParams:
    # number of injected attack EVENTS over the 60-day horizon (all machines)
    mitm_fdi: int = 800       # payload modified in transit (no key)
    replay: int = 800
    spoof: int = 400
    timestamp: int = 500      # time field altered in transit
    delay: int = 500          # unmodified packets held and released late
    dos: int = 300
    source_fdi: int = 150     # signed false data (compromised node / CT tamper)
    db_tamper: int = 500      # stored rows altered on the database host
    combined: int = 100       # replay masking a real overload
    poisoning: int = 8        # slow drift by a compromised node
    # parameter ranges
    mitm_scale: tuple = (0.70, 0.98)
    dur_min: tuple = (1, 30)
    replay_age_h: tuple = (1, 24)
    delay_s: tuple = (2, 600)
    dos_s: tuple = (10, 1800)
    src_fdi_scale: tuple = (0.01, 0.15)   # fractional under-reporting
    src_fdi_min: tuple = (30, 240)
    src_fdi_physical_frac: float = 0.5    # share that keeps P = sqrt3 V I PF
    poison_rate_per_day: float = 0.005
    poison_days: tuple = (10, 20)

PLANT = PlantParams()
EVENTS = EventParams()
NET = NetParams()
ATTACK = AttackParams()

# --- economics (Section 10.4) ------------------------------------------------
CAPEX_INR = 1_690_000
TARIFF_INR_KWH = 8.5
DEMAND_INR_KVA_MONTH = 400.0   # assumed billed maximum-demand charge
MAINT_FRAC = 0.05
DISCOUNT = 0.08
LIFE_YEARS = 5
