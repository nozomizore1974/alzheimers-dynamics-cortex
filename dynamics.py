"""
dynamics.py -- structural + mean-field stability analysis of one column.

Answers, *without running the spiking simulator*, whether the network's
asynchronous-irregular (AI) state is expected to be stable, or whether the
column is driven toward a synchronous-regular / oscillatory (SR) state -- and,
if so, through which populations and at what frequency. Three tiers, all built
from the derived model (:class:`model_params.ModelParams`): synapse in-degrees
``indeg`` (K), signed weights ``W`` (pA), neuron numbers and neuron parameters,
external drive and local delays.

  * Tier 0 -- wiring only (no rates, no neuron dynamics):
      - ``ei_balance``            g_eff = A_I / A_E per population (Brunel's g,
                                  carried by the signed weights)
      - ``input_moments``         mean/fluctuation drive mu, sigma and mu/sigma
      - ``effective_connectivity``/``structural_gain`` -- M = K.W and its
                                  spectral radius (recurrent-gain proxy)
  * Tier 1 -- mean field at the operating rates (nnmt LIF transfer function):
      - ``working_point``         (V_th - mu)/sigma per population; <=0 mean-
                                  driven (SR risk), large fluctuation-driven (AI)
  * Tier 2 -- linear stability of the async state (Bos et al. 2016):
      - ``oscillation_spectrum``  eigenvalues of the effective connectivity vs
                                  frequency; margin = 1 - max Re(lambda) (<=0
                                  unstable), with the critical frequency.

Caveat: the mean-field transfer function is the LIF (iaf) one. For the ``aeif``
model family it is used as an approximation -- the exponential spike mechanism
and the adaptation current (a, b, tau_w) are neglected. Treat the *sign* of the
stability margin and the *ordering* of populations as robust; absolute values
are approximate. Requires ``nnmt``.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from model_params import tau_m_of

_EXC_C, _INH_C = "#1f77b4", "#d62728"


# ── neuron / mean-field parameters from the model ─────────────────────────────

def _neuron_arrays(mp):
    """Per-(target) population neuron parameters as numpy arrays (SI-ish units).

    Uses the healthy E/I parameter sets (``spec['E_healthy']``/``['I_healthy']``)
    picked per population by ``mp.is_exc``. Voltages are relative to the leak
    potential E_L (mV); times in seconds.
    """
    E, I = mp.spec["E_healthy"], mp.spec["I_healthy"]
    is_e = np.asarray(mp.is_exc)
    pick = lambda k: np.array([(E if e else I)[k] for e in is_e], dtype=float)
    tau_m = np.array([tau_m_of(E if e else I) for e in is_e]) * 1e-3      # s
    C_m = pick("C_m")                                                    # pF
    t_ref = pick("t_ref") * 1e-3                                         # s
    V_th = (pick("V_th") - pick("E_L"))                                  # mV rel E_L
    V_0 = (pick("V_reset") - pick("E_L"))                               # mV rel E_L
    tau_ex = pick("tau_syn_ex") * 1e-3                                   # s
    tau_in = pick("tau_syn_in") * 1e-3                                   # s
    return dict(tau_m=tau_m, C_m=C_m, t_ref=t_ref, V_th=V_th, V_0=V_0,
                tau_ex=tau_ex, tau_in=tau_in, is_e=is_e)


def _mean_field_inputs(mp):
    """Assemble the nnmt mean-field input parameters (weights in mV).

    Mirrors ``theory/rates.py`` in the Adnest project: the PSC weight ``W`` (pA)
    is turned into the DC voltage gain ``J = W * tau_syn / C_m`` (mV), with the
    synaptic time constant of the *target*'s receptor (ex for an E source, in for
    an I source). The external Poisson drive contributes one connection
    (``K_ext = 1``) per population at rate ``rate_ext``.
    """
    na = _neuron_arrays(mp)
    n = len(mp.pop_names)
    is_e = na["is_e"]
    # synaptic tau per (target i, source j): target's ex tau for E source else in
    tau_syn = np.where(is_e[None, :], na["tau_ex"][:, None], na["tau_in"][:, None])
    J = mp.W * 1e3 * tau_syn / na["C_m"][:, None]           # mV
    K = np.asarray(mp.indeg, dtype=float)
    J_ext = mp.w_ext * 1e3 * na["tau_ex"] / na["C_m"]        # mV (external is exc)
    input_prms = dict(tau_m=na["tau_m"], J=J, K=K,
                      J_ext=np.diag(J_ext), K_ext=np.diag(np.ones(n)),
                      nu_ext=np.asarray(mp.rate_ext, dtype=float))
    return input_prms, tau_syn, na


def _rates_array(mp, rates):
    """Coerce a firing-rate spec (dict {pop: Hz} or array) to an (n,) array."""
    if isinstance(rates, dict):
        return np.array([rates.get(p, 0.0) for p in mp.pop_names], dtype=float)
    r = np.asarray(rates, dtype=float)
    if r.shape[0] != len(mp.pop_names):
        raise ValueError("rates length does not match number of populations")
    return r


# ── Tier 0: structural indicators ─────────────────────────────────────────────

def ei_balance(mp):
    """Structural effective E/I input balance g_eff = A_I/A_E per population.

    ``A_x = sum_{j in x} K[i,j] |W[i,j]|`` over excitatory / inhibitory sources.
    ``>1`` inhibition-dominated, ``<1`` excitation-dominated (SR risk).
    """
    K, W, is_e = np.asarray(mp.indeg), np.asarray(mp.W), np.asarray(mp.is_exc)
    C = K * W
    exc = C[:, is_e].sum(axis=1)
    inh = -C[:, ~is_e].sum(axis=1)          # I-source weights are negative
    with np.errstate(divide="ignore", invalid="ignore"):
        return inh / np.where(exc != 0, exc, np.nan)


def input_moments(mp):
    """Structural recurrent mean/fluctuation per population.

    Returns dict with ``mu = sum_j K W`` (net signed drive), ``sigma =
    sqrt(sum_j K W^2)`` and ``drive_index = mu/sigma`` (>0 net excitatory,
    mean-dominated -> SR tendency).
    """
    K, W = np.asarray(mp.indeg), np.asarray(mp.W)
    mu = (K * W).sum(axis=1)
    sigma = np.sqrt((K * W ** 2).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        idx = mu / np.where(sigma != 0, sigma, np.nan)
    return dict(mu=mu, sigma=sigma, drive_index=idx)


def effective_connectivity(mp):
    """Signed effective connectivity ``M = K . W`` (target x source)."""
    return np.asarray(mp.indeg) * np.asarray(mp.W)


def structural_gain(M):
    """Spectral radius (largest |eigenvalue|) of an effective-connectivity matrix."""
    ev = np.linalg.eigvals(np.nan_to_num(np.asarray(M, dtype=float)))
    return float(np.max(np.abs(ev))), ev


# ── Tier 1: mean-field working point ──────────────────────────────────────────

def working_point(mp, rates):
    """Mean-field working point ``(V_th - mu)/sigma`` per population at ``rates``.

    ``mu``/``sigma`` (mV) are the nnmt LIF mean/std of the input at the given
    firing rates. ``> ~1`` fluctuation-driven (AI); ``<= 0`` mean-driven (SR).
    """
    from nnmt.lif._general import _mean_input, _std_input
    input_prms, _, na = _mean_field_inputs(mp)
    nu = _rates_array(mp, rates)
    mu = np.asarray(_mean_input(nu=nu, **input_prms))
    sigma = np.asarray(_std_input(nu=nu, **input_prms))
    with np.errstate(divide="ignore", invalid="ignore"):
        wp = (na["V_th"] - mu) / np.where(sigma != 0, sigma, np.nan)
    return dict(mu=mu, sigma=sigma, working_point=wp)


# ── Tier 2: oscillatory-instability spectrum ──────────────────────────────────

def _delay_matrices(mp):
    """Local delay mean/sd (s), target x source, from the model's delays."""
    is_e = np.asarray(mp.is_exc)
    delay = np.where(is_e[None, :], mp.delay_e, mp.delay_i) * 1e-3   # s, by source
    delay = np.broadcast_to(delay, (len(is_e), len(is_e))).copy()
    return delay, mp.delay_rel * delay


def oscillation_spectrum(mp, rates, freqs=None, delay_dist="truncated_gaussian"):
    """Effective-connectivity eigenvalue spectrum vs frequency (Bos et al. 2016).

    Builds ``M_eff(w) = tau_m * J * K * H(w) * D(w)`` at the working point set by
    ``rates`` and returns its eigenvalues over frequency. The async state is
    unstable when an eigenvalue reaches ``1+0j``:
    ``margin = 1 - max Re(lambda) <= 0``.

    Returns dict: ``freqs``, ``eigenvalues`` (n_freq, n_pop), ``max_real``,
    ``critical_freq``, ``peak``, ``margin``.
    """
    from nnmt.lif._general import _mean_input, _std_input
    from nnmt.lif.exp import _transfer_function, _effective_connectivity
    from nnmt.network_properties import _delay_dist_matrix

    if freqs is None:
        freqs = np.linspace(1.0, 300.0, 300)
    input_prms, tau_syn, na = _mean_field_inputs(mp)
    nu = _rates_array(mp, rates)

    mu_V = np.asarray(_mean_input(nu=nu, **input_prms)) * 1e-3
    sig_V = np.asarray(_std_input(nu=nu, **input_prms)) * 1e-3
    V_th_V, V_0_V = na["V_th"] * 1e-3, na["V_0"] * 1e-3
    J_V = input_prms["J"] * 1e-3
    K = input_prms["K"]
    tau_m = float(np.mean(na["tau_m"]))            # uniform in this model
    tau_s = float(np.max(tau_syn))
    t_ref = na["t_ref"]

    omegas = 2 * np.pi * freqs
    tf = _transfer_function(mu_V, sig_V, tau_m, tau_s, t_ref, V_th_V, V_0_V, omegas)
    delay, delay_sd = _delay_matrices(mp)
    D = _delay_dist_matrix(delay, delay_sd, delay_dist, omegas)
    D = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
    M_eff = _effective_connectivity(tf, D, J_V, K, tau_m)
    M_eff = np.nan_to_num(M_eff, nan=0.0, posinf=0.0, neginf=0.0)

    eig = np.array([np.linalg.eigvals(M_eff[i]) for i in range(len(freqs))])
    max_real = eig.real.max(axis=1)
    ic = int(np.argmax(max_real))
    peak = float(max_real[ic])
    return dict(freqs=freqs, eigenvalues=eig, max_real=max_real,
                critical_freq=float(freqs[ic]), peak=peak, margin=1.0 - peak)


# ── plotting (cortex style: savefig dpi=120, close) ───────────────────────────

def _bar_colors(mp):
    return [_EXC_C if e else _INH_C for e in mp.is_exc]


def plot_structural(mp, path):
    """Tier-0 figure: g_eff, drive index, and the effective-connectivity matrix."""
    g = ei_balance(mp)
    mom = input_moments(mp)
    M = effective_connectivity(mp)
    rho, _ = structural_gain(M)
    names, colors = mp.pop_names, _bar_colors(mp)
    x = np.arange(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].bar(x, g, color=colors)
    axes[0].axhline(1.0, color="k", ls="--", lw=1)
    axes[0].set_title("E/I input balance  g_eff = A_I/A_E")
    axes[0].set_ylabel("g_eff  (<1 exc-dominated)")

    axes[1].bar(x, mom["drive_index"], color=colors)
    axes[1].axhline(0.0, color="k", ls="--", lw=1)
    axes[1].set_title("mean-drive index  mu/sigma  (>0 mean-driven)")

    v = np.max(np.abs(M)) or 1.0
    im = axes[2].imshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-v, vcenter=0, vmax=v))
    axes[2].set_xticks(x); axes[2].set_xticklabels(names, rotation=90, fontsize=7)
    axes[2].set_yticks(x); axes[2].set_yticklabels(names, fontsize=7)
    axes[2].set_title(f"effective connectivity K.W  (rho={rho:.2g})")
    axes[2].set_xlabel("source"); axes[2].set_ylabel("target")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes[:2]:
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=90, fontsize=8)
        ax.grid(axis="y", ls="--", lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    fig.suptitle("Tier 0 -- structural indicators (wiring only)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return dict(g_eff=dict(zip(names, g)), spectral_radius=rho)


def plot_working_point(mp, rates, path, threshold=1.0):
    """Tier-1 figure: working point (V_th - mu)/sigma per population."""
    wp = working_point(mp, rates)
    names, colors = mp.pop_names, _bar_colors(mp)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x, wp["working_point"], color=colors)
    ax.axhline(threshold, color="red", ls="--", lw=1.2,
               label=f"AI/SR heuristic (={threshold:g})")
    ax.axhline(0.0, color="k", ls="-", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_ylabel("(V_th - mu)/sigma")
    ax.set_title("Tier 1 -- mean-field working point  (<= 0 mean-driven -> SR)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", ls="--", lw=0.4, alpha=0.5); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return {n: float(v) for n, v in zip(names, wp["working_point"])}


def plot_oscillation_stability(mp, rates, path, freqs=None):
    """Tier-2 figure: max Re(lambda) vs frequency + eigenvalue loci."""
    spec = oscillation_spectrum(mp, rates, freqs=freqs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].plot(spec["freqs"], spec["max_real"], color="#1f77b4", lw=1.6)
    axes[0].axhline(1.0, color="red", ls="--", lw=1.2, label="instability (=1)")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("max Re lambda  of effective connectivity")
    axes[0].set_title(f"margin={spec['margin']:.2f}, f*={spec['critical_freq']:.0f} Hz")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, ls="--", lw=0.4, alpha=0.5)

    ev = spec["eigenvalues"].ravel()
    fr = np.repeat(spec["freqs"][:, None], spec["eigenvalues"].shape[1], axis=1).ravel()
    sc = axes[1].scatter(ev.real, ev.imag, c=fr, s=4, cmap="viridis", alpha=0.5)
    axes[1].axvline(1.0, color="red", ls="--", lw=0.8)
    axes[1].plot(1, 0, "rx", ms=9, mew=2)
    axes[1].set_xlabel("Re lambda"); axes[1].set_ylabel("Im lambda")
    axes[1].set_title("eigenvalues (colour = Hz)")
    axes[1].grid(True, ls="--", lw=0.4, alpha=0.5)
    fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04, label="Hz")

    fig.suptitle("Tier 2 -- asynchronous-state stability", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return dict(margin=spec["margin"], critical_freq=spec["critical_freq"],
                peak=spec["peak"])


# ── runner ────────────────────────────────────────────────────────────────────

def analyze(mp, rates, outdir, prefix="dynamics"):
    """Compute all three tiers and write the figures into ``outdir``.

    Parameters
    ----------
    mp : model_params.ModelParams
    rates : dict {pop: Hz} or array   operating firing rates (e.g. from a run's
        ``cache/rates.pkl`` -> ``value[0]``, or ``analysis.firing_rates``).
    outdir : str
    Returns a metrics dict (g_eff, spectral radius, working points, stability
    margin / critical frequency).
    """
    os.makedirs(outdir, exist_ok=True)
    m = {}
    m["structural"] = plot_structural(mp, os.path.join(outdir, f"{prefix}_structural.png"))
    m["working_point"] = plot_working_point(mp, rates,
                                            os.path.join(outdir, f"{prefix}_working_point.png"))
    m["stability"] = plot_oscillation_stability(mp, rates,
                                                os.path.join(outdir, f"{prefix}_stability.png"))
    return m


def load_run(exp, config_path=None):
    """Rebuild ``mp`` and load a run's cached population rates for analysis.

    Convenience for analysing an existing ``out/<exp>/`` without re-simulating.
    The config is taken (in order of preference) from ``config_path``, the run's
    own ``out/<exp>/config_used.json`` (written by ``main.py``), or the default
    ``config.yaml`` -- so a run analysed this way uses the parameters that
    actually produced it whenever that snapshot is present.
    """
    import json
    import pickle
    import data_loader
    import model_params as MP
    here = os.path.dirname(os.path.abspath(__file__))
    run_cfg = os.path.join(here, "out", exp, "config_used.json")
    if config_path is not None:
        cfg = data_loader.load_config(config_path)
    elif os.path.exists(run_cfg):
        with open(run_cfg) as f:
            cfg = json.load(f)
    else:
        cfg = data_loader.load_config()
    mc = data_loader.load_microcircuit()
    nps = data_loader.load_neuron_params()
    syn = cfg["synapse"]
    spec = data_loader.build_spec(nps, cfg["model"],
                                  tau_syn_ex=syn.get("tau_syn_ex"),
                                  tau_syn_in=syn.get("tau_syn_in"))
    mp = MP.compute_model_params(mc, spec, cfg)
    with open(os.path.join(here, "out", exp, "cache", "rates.pkl"), "rb") as f:
        pop_rates = pickle.load(f)["value"][0]
    return mp, pop_rates
