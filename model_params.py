"""
model_params.py -- compute derived model parameters for one column.

From the microcircuit definition + neuron spec + config it builds, target x
source (8x8):
  * n_syn   : total synapse numbers (Potjans & Diesmann fixed-total-number rule)
  * indeg   : indegree per target neuron (n_syn / N_target)
  * W        : signed mean synaptic weight (pA), incl. PSP->PSC conversion, the
              linear indegree (scale) correction, the L4E->L2/3E doubling and
              the (load-dependent) disinhibition on I->E.
plus per-population external Poisson weights/rates calibrated to threshold.

Self-contained: numpy only. The PSP->PSC conversion matches the standard
exponential-current formula (singularity at tau_syn == tau_m handled by its
analytic limit e).
"""
from dataclasses import dataclass, field
import numpy as np


def tau_m_of(p):
    """Membrane time constant tau_m = R_m * C_m (ms).

    Two neuron parameterisations coexist in the spec and report tau_m differently:
      * iaf_psc_exp exposes ``tau_m`` directly.
      * aeif_* exposes capacitance ``C_m`` (pF) and leak conductance ``g_L`` (nS).
        Since the membrane resistance is R_m = 1 / g_L, we have tau_m = C_m / g_L.
    tau_m sets how fast the membrane integrates and forgets input; it enters every
    PSP<->PSC conversion and every external-rate calibration below.
    """
    return p["tau_m"] if "tau_m" in p else p["C_m"] / p["g_L"]


def psc_over_psp(p, syn="ex"):
    """Peak-PSC (pA) produced per unit peak-PSP (mV) for an exponential current synapse.

    A presynaptic spike injects the current  I(t) = J * exp(-t / tau_s), where J is
    the synaptic weight (pA) and tau_s = tau_syn. For a leaky membrane

        C_m dV/dt = -g_L V + I(t),     tau_m = C_m / g_L,

    the voltage response is a difference of two exponentials that peaks at

        t_peak = (tau_m * tau_s) / (tau_m - tau_s) * ln(tau_m / tau_s).

    Writing  eps := tau_s / tau_m  and substituting t_peak collapses the peak
    deflection to the compact form

        V_peak = (J * tau_m / C_m) * eps ** ( 1 / (1 - eps) ),

    so the inverse conversion -- weight (pA) needed per millivolt of PSP -- is

        J / V_peak = (C_m / tau_m) * eps ** ( -1 / (1 - eps) ).

    Connectivity is specified in terms of target PSP amplitudes (mV); multiplying
    by this factor turns them into the PSC weights (pA) the simulator expects.
    The closed form is singular at tau_s == tau_m (eps == 1); its analytic limit
    there is Euler's number e, returned explicitly to avoid 0/0.
    """
    C, tau, ts = p["C_m"], tau_m_of(p), p["tau_syn_" + syn]
    eps = ts / tau
    factor = np.e if abs(1.0 - eps) < 1e-6 else eps ** (-1.0 / (1.0 - eps))
    return C * factor / tau


def num_synapses(p, n_pre, n_post):
    """Total synapse count realising a pairwise connection probability ``p``.

    Following Potjans & Diesmann (2014), synapses are drawn one at a time, each
    independently picking a (source, target) pair uniformly from the n_pre * n_post
    possible pairs, *with replacement* (so multiple contacts between the same pair --
    multapses -- are allowed). After K draws the probability that a given pair has
    received at least one synapse is

        P_conn = 1 - (1 - 1/(n_pre * n_post)) ** K.

    Setting P_conn = p and solving for K gives the fixed-total-number rule

        K = ln(1 - p) / ln(1 - 1/(n_pre * n_post)),

    rounded to the nearest integer. This pins the expected pairwise probability to p
    while letting the per-pair contact count fluctuate naturally. p <= 0 -> no
    connection (K = 0).
    """
    if p <= 0.0:
        return 0
    return int(round(np.log(1.0 - p) / np.log(1.0 - 1.0 / (n_pre * n_post))))


def wIE_factor(load, g_inh, wIE_floor):
    """Effective inhibitory->excitatory (I->E) weight scaling under amyloid-beta load.

    Two multiplicative effects act on I->E synapses:
      * ``g_inh`` -- a fixed global inhibition gain applied at every load.
      * load-dependent disinhibition -- as the Abeta load fraction ``load`` (0..1)
        grows, inhibition onto E cells is progressively withdrawn, interpolating
        linearly from full strength at load = 0 down to a residual floor
        ``wIE_floor`` at load = 1:

            factor = g_inh * (1 - load * (1 - wIE_floor)).

    This is the model's mechanistic knob for Abeta-induced loss of inhibition, a
    leading candidate driver of cortical hyperexcitability in Alzheimer's disease.
    """
    return g_inh * (1.0 - load * (1.0 - wIE_floor))


@dataclass
class ModelParams:
    pop_names: list
    layers: list
    is_exc: np.ndarray
    pop_layer: list
    N: np.ndarray                 # scaled population sizes
    conn_probs: np.ndarray
    n_syn: np.ndarray             # (8,8) target x source
    indeg: np.ndarray             # (8,8)
    W: np.ndarray                 # (8,8) signed mean weight pA, target x source
    w_ext: np.ndarray             # (8,) external weight pA per population
    rate_ext: np.ndarray          # (8,) external Poisson rate Hz per population
    tau_syn: np.ndarray           # (8,) per-source synaptic tau (ex for E, in for I)
    delay_e: float
    delay_i: float
    delay_rel: float
    PSP_rel: float
    spec: dict = field(repr=False)
    abeta_ratio: float = 0.0


def compute_model_params(mc, spec, cfg):
    """
    Build a :class:`ModelParams` for a single column at load ``abeta_ratio``.

    Parameters
    ----------
    mc : dict        microcircuit (from data_loader.load_microcircuit)
    spec : dict      neuron spec (from data_loader.build_spec)
    cfg : dict       run config (from data_loader.load_config)
    """
    pop_names = mc["pop_names"]
    is_exc = mc["is_exc"]
    conn_probs = mc["conn_probs"]
    n = len(pop_names)

    from data_loader import scaled_neuron_numbers
    N = scaled_neuron_numbers(mc["N_full"], cfg["N_scale"])

    syn = cfg["synapse"]
    dly = cfg["delay"]
    load = float(cfg["abeta_ratio"])

    # --- PSP -> PSC conversion factors ---
    # The conversion depends on whose membrane integrates the current (the *target*
    # neuron) and on the synapse kind (excitatory "ex" for an E source, inhibitory
    # "in" for an I source). Hence four factors: target {E,I} x synapse {ex,in}.
    cE_ex = psc_over_psp(spec["E_healthy"], "ex")   # E target, EPSC
    cE_in = psc_over_psp(spec["E_healthy"], "in")   # E target, IPSC
    cI_ex = psc_over_psp(spec["I_healthy"], "ex")   # I target, EPSC
    cI_in = psc_over_psp(spec["I_healthy"], "in")   # I target, IPSC
    PSP_e, g = syn["PSP_e"], syn["g"]               # reference EPSP (mV); g = IPSP/EPSP ratio (<0)
    # Linear indegree (scale) correction: downscaling the network by N_scale cuts
    # each neuron's indegree ~proportionally, so synaptic weights are scaled up by
    # 1/N_scale (relative to the reference scale at which PSP_e was tuned) to keep
    # the total recurrent input current per neuron -- and thus the dynamics -- invariant.
    kfac = cfg["weight_ref_scale"] / cfg["N_scale"]

    # Signed mean weight (pA) for each (source-type, target-type) pair:
    #   weight_pA = kfac * (target/synapse conversion) * (g for I sources) * PSP_e.
    # E sources carry the +PSP_e EPSP; I sources carry g*PSP_e, where g<0 makes the
    # current inhibitory and |g| sets the IPSP/EPSP amplitude ratio.
    base_W = {
        ("E", "E"): kfac * cE_ex * PSP_e,    ("E", "I"): kfac * cI_ex * PSP_e,
        ("I", "E"): kfac * cE_in * g * PSP_e, ("I", "I"): kfac * cI_in * g * PSP_e,
    }
    # I->E weights are additionally scaled by the (Abeta-load-dependent) disinhibition factor.
    wIE = wIE_factor(load, syn["g_inh"], syn["wIE_floor"])

    # --- target x source matrices ---
    n_syn = np.zeros((n, n), dtype=int)
    indeg = np.zeros((n, n))
    W = np.zeros((n, n))
    # Row i = target population, column j = source population (target x source).
    for i, tgt in enumerate(pop_names):
        for j, src in enumerate(pop_names):
            # Total contacts realising conn_probs[i,j], then indegree = contacts / N_target.
            k = num_synapses(conn_probs[i, j], N[j], N[i])
            n_syn[i, j] = k
            indeg[i, j] = k / N[i]
            st = "E" if is_exc[j] else "I"   # source type
            tt = "E" if is_exc[i] else "I"   # target type
            w = base_W[(st, tt)]
            # P&D fix: the L4E -> L2/3E projection is empirically ~2x stronger.
            if src == "L4E" and tgt == "L23E":
                w *= syn["local_scaling4Eto23E"]
            # Apply global + Abeta-load disinhibition only to inhibitory->excitatory weights.
            if st == "I" and tt == "E":
                w *= wIE
            W[i, j] = w

    # --- per-source synaptic tau (for LFP kernel: ex for E source, in for I) ---
    tau_syn = np.array([
        (spec["E_healthy"] if is_exc[j] else spec["I_healthy"])[
            "tau_syn_ex" if is_exc[j] else "tau_syn_in"] for j in range(n)])

    # --- external Poisson drive, calibrated to a target mean depolarisation ---
    # Each population receives independent excitatory Poisson input. We pick its rate
    # so the mean voltage it drives equals  eta_ext * (V_th - E_L), i.e. eta_ext is the
    # fraction of the rheobase distance covered by background drive alone.
    #
    # An exp-current synapse delivers charge w*tau_s per spike; at rate r the mean
    # current is r*w*tau_s, and a leaky membrane (R_m = tau_m / C_m) turns that into a
    # steady-state voltage  V = r * w * tau_s * tau_m / C_m. Setting V = eta*(V_th-E_L)
    # and solving for r gives the expression below; the 1e3 converts kHz (tau in ms) to Hz.
    eta = cfg["input"]["eta_ext"]
    PSP_ext = syn["PSP_ext"]                          # external EPSP amplitude (mV)
    eN, iN = spec["E_healthy"], spec["I_healthy"]
    w_ext_E, w_ext_I = cE_ex * PSP_ext, cI_ex * PSP_ext   # -> external weights (pA)
    rE = 1e3 * (eN["V_th"] - eN["E_L"]) * eta * eN["C_m"] / (tau_m_of(eN) * eN["tau_syn_ex"] * w_ext_E)
    rI = 1e3 * (iN["V_th"] - iN["E_L"]) * eta * iN["C_m"] / (tau_m_of(iN) * iN["tau_syn_ex"] * w_ext_I)
    w_ext = np.zeros(n)
    rate_ext = np.zeros(n)
    for j, name in enumerate(pop_names):
        if is_exc[j]:
            # L5E/L6E get population-specific external-weight tweaks (P&D layer corrections).
            w_ext[j] = w_ext_E * (syn["scaling5E"] if name == "L5E"
                                  else syn["scaling6E"] if name == "L6E" else 1.0)
            rate_ext[j] = rE
        else:
            w_ext[j] = w_ext_I
            rate_ext[j] = rI

    return ModelParams(
        pop_names=pop_names, layers=mc["layers"], is_exc=is_exc, pop_layer=mc["pop_layer"],
        N=N, conn_probs=conn_probs, n_syn=n_syn, indeg=indeg, W=W,
        w_ext=w_ext, rate_ext=rate_ext, tau_syn=tau_syn,
        delay_e=dly["delay_e"] * dly["delay_scale"], delay_i=dly["delay_i"] * dly["delay_scale"],
        delay_rel=dly["delay_rel"], PSP_rel=syn["PSP_rel"], spec=spec, abeta_ratio=load)
