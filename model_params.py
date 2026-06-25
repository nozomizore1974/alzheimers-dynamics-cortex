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
    """Membrane time constant: explicit for iaf, C_m/g_L for aeif."""
    return p["tau_m"] if "tau_m" in p else p["C_m"] / p["g_L"]


def psc_over_psp(p, syn="ex"):
    """Conversion factor PSP (mV) -> PSC (pA) for exponential synaptic currents."""
    C, tau, ts = p["C_m"], tau_m_of(p), p["tau_syn_" + syn]
    eps = ts / tau
    factor = np.e if abs(1.0 - eps) < 1e-6 else eps ** (-1.0 / (1.0 - eps))
    return C * factor / tau


def num_synapses(p, n_pre, n_post):
    """Total synapses realising pairwise connection probability p (multapses allowed)."""
    if p <= 0.0:
        return 0
    return int(round(np.log(1.0 - p) / np.log(1.0 - 1.0 / (n_pre * n_post))))


def wIE_factor(load, g_inh, wIE_floor):
    """Effective I->E scaling: global reduction x load-dependent disinhibition."""
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

    # --- PSP -> PSC conversion factors (target neuron x synapse type) ---
    cE_ex = psc_over_psp(spec["E_healthy"], "ex")
    cE_in = psc_over_psp(spec["E_healthy"], "in")
    cI_ex = psc_over_psp(spec["I_healthy"], "ex")
    cI_in = psc_over_psp(spec["I_healthy"], "in")
    PSP_e, g = syn["PSP_e"], syn["g"]
    kfac = cfg["weight_ref_scale"] / cfg["N_scale"]    # linear indegree correction

    base_W = {
        ("E", "E"): kfac * cE_ex * PSP_e,    ("E", "I"): kfac * cI_ex * PSP_e,
        ("I", "E"): kfac * cE_in * g * PSP_e, ("I", "I"): kfac * cI_in * g * PSP_e,
    }
    wIE = wIE_factor(load, syn["g_inh"], syn["wIE_floor"])

    # --- target x source matrices ---
    n_syn = np.zeros((n, n), dtype=int)
    indeg = np.zeros((n, n))
    W = np.zeros((n, n))
    for i, tgt in enumerate(pop_names):
        for j, src in enumerate(pop_names):
            k = num_synapses(conn_probs[i, j], N[j], N[i])
            n_syn[i, j] = k
            indeg[i, j] = k / N[i]
            st = "E" if is_exc[j] else "I"
            tt = "E" if is_exc[i] else "I"
            w = base_W[(st, tt)]
            if src == "L4E" and tgt == "L23E":
                w *= syn["local_scaling4Eto23E"]
            if st == "I" and tt == "E":
                w *= wIE
            W[i, j] = w

    # --- per-source synaptic tau (for LFP kernel: ex for E source, in for I) ---
    tau_syn = np.array([
        (spec["E_healthy"] if is_exc[j] else spec["I_healthy"])[
            "tau_syn_ex" if is_exc[j] else "tau_syn_in"] for j in range(n)])

    # --- external Poisson drive, calibrated to eta_ext * (V_th - E_L) ---
    eta = cfg["input"]["eta_ext"]
    PSP_ext = syn["PSP_ext"]
    eN, iN = spec["E_healthy"], spec["I_healthy"]
    w_ext_E, w_ext_I = cE_ex * PSP_ext, cI_ex * PSP_ext
    rE = 1e3 * (eN["V_th"] - eN["E_L"]) * eta * eN["C_m"] / (tau_m_of(eN) * eN["tau_syn_ex"] * w_ext_E)
    rI = 1e3 * (iN["V_th"] - iN["E_L"]) * eta * iN["C_m"] / (tau_m_of(iN) * iN["tau_syn_ex"] * w_ext_I)
    w_ext = np.zeros(n)
    rate_ext = np.zeros(n)
    for j, name in enumerate(pop_names):
        if is_exc[j]:
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
