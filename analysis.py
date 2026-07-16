"""
analysis.py -- results analysis for one column simulation.

Provides: firing rates, binned spike counts (total / per-layer / per-population,
PSTH-style), kernel LFP proxy (ROI + per layer), power spectral density
(Welch / FFT), Morlet wavelet scalogram, and cross-frequency theta-beta
phase-locking value (a layer x layer n:m PLV matrix and a fine-frequency
comodulogram).

Self-contained: numpy + scipy only.
"""
import numpy as np
from scipy.signal import convolve, welch, hilbert, butter, sosfiltfilt


# --------------------------------------------------------------------------- #
#  PSC kernel + firing rates                                                   #
# --------------------------------------------------------------------------- #
def kernel_for_psc(tau_s, dt):
    """Causal, normalised exponential PSC kernel."""
    t_ker = np.arange(-10 * tau_s, 10 * tau_s, dt)
    kernel = np.exp(-t_ker / tau_s) / tau_s
    kernel[t_ker < -dt / 2] = 0.0
    kernel /= kernel.sum()
    return kernel


def _spike_counts(result, g0, g1, t_edges):
    s, t = result["senders"], result["times"]
    m = (s >= g0) & (s <= g1)
    counts, _ = np.histogram(t[m], bins=t_edges)
    return counts


def firing_rates(result, dt=None):
    """
    Mean firing rate (Hz) per population and per subset (after warmup).

    Returns (pop_rates: dict, subset_rates: list-of-dict).
    """
    s, t = result["senders"], result["times"]
    warm, tsim = result["warmup"], result["t_sim"]
    dur = (tsim - warm) / 1000.0
    pop_rates = {}
    for name, (g0, g1, n) in result["pop_gid"].items():
        cnt = int(((s >= g0) & (s <= g1) & (t > warm)).sum())
        pop_rates[name] = cnt / (n * dur)
    subset_rates = []
    for r in result["subsets"]:
        cnt = int(((s >= r["g0"]) & (s <= r["g1"]) & (t > warm)).sum())
        subset_rates.append({**r, "rate": cnt / (r["n"] * dur)})
    return pop_rates, subset_rates


def spike_count_series(result, bin_ms=1.0):
    """
    Total spike count per time bin across all recorded neurons (PSTH-style),
    over the full ``t_sim`` span -- pairs with a raster plot as a density strip.

    Returns (t_centres (ms), counts) with bin width ``bin_ms``.
    """
    t_sim = result["t_sim"]
    n_bins = max(1, int(round(t_sim / bin_ms)))
    edges = np.linspace(0.0, t_sim, n_bins + 1)
    counts, _ = np.histogram(result["times"], bins=edges)
    centres = (edges[:-1] + edges[1:]) / 2.0
    return centres, counts


def population_spike_count_series(result, bin_ms=1.0):
    """
    Per-population spike count per time bin (the model's 8 populations), on the
    same bins as :func:`spike_count_series` -- pairs with a sub-population raster.

    Returns (t_centres (ms), {pop_name: counts}).
    """
    t_sim = result["t_sim"]
    n_bins = max(1, int(round(t_sim / bin_ms)))
    edges = np.linspace(0.0, t_sim, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2.0
    counts = {name: _spike_counts(result, g0, g1, edges)
              for name, (g0, g1, _n) in result["pop_gid"].items()}
    return centres, counts


def layer_spike_count_series(result, mp, bin_ms=1.0):
    """
    Per-layer spike count per time bin, summing the populations that share a
    layer (built on :func:`population_spike_count_series`).

    Returns (t_centres (ms), {layer: counts}).
    """
    centres, pop_counts = population_spike_count_series(result, bin_ms)
    layer_counts = {
        lay: sum(pop_counts[name] for name, pl in zip(mp.pop_names, mp.pop_layer) if pl == lay)
        for lay in mp.layers
    }
    return centres, layer_counts


def health_spike_count_series(result, bin_ms=1.0):
    """
    Total spike count per time bin split by health status: 'healthy' subsets
    vs all Abeta-affected subsets (abeta_hyper/abeta_supp/abeta pooled into
    'abeta'), on the same bins as :func:`spike_count_series`.

    Returns (t_centres (ms), {"healthy": counts, "abeta": counts}).
    """
    t_sim = result["t_sim"]
    n_bins = max(1, int(round(t_sim / bin_ms)))
    edges = np.linspace(0.0, t_sim, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2.0
    groups = {"healthy": np.zeros(n_bins, dtype=int), "abeta": np.zeros(n_bins, dtype=int)}
    for r in result["subsets"]:
        key = "healthy" if r["kind"] == "healthy" else "abeta"
        groups[key] += _spike_counts(result, r["g0"], r["g1"], edges)
    return centres, groups


# --------------------------------------------------------------------------- #
#  LFP proxy (kernel method): ROI, per-layer, per-population                   #
# --------------------------------------------------------------------------- #
def lfp_proxy(result, mp, resolution):
    """
    Kernel LFP proxy. Per-population rate -> PSC-kernel convolution -> signed
    weight x indegree projection -> I_ex + 1.65*I_in per target population ->
    NN-weighted aggregation.

    Returns
    -------
    dict with: t (ms), roi (1d), roi_z (1d, z-scored roi), layer ({layer: 1d}),
    pop ((8, T)).
    """
    dt = resolution
    n_bins = int(round(result["t_sim"] / dt))
    t_edges = np.linspace(0.0, result["t_sim"], n_bins + 1)
    t_axis = np.linspace(dt, result["t_sim"], n_bins)

    n = len(mp.pop_names)
    P = np.zeros((n, n_bins))
    for j, name in enumerate(mp.pop_names):
        g0, g1, nn = result["pop_gid"][name]
        rate = _spike_counts(result, g0, g1, t_edges).astype(float) / (nn * dt * 1e-3)
        P[j] = convolve(mp.tau_syn[j] * rate, kernel_for_psc(mp.tau_syn[j], dt), mode="same")

    ws = mp.W * mp.indeg                          # signed weight x indegree
    I_ex = np.clip(ws, 0, None) @ P
    I_in = np.clip(ws, None, 0) @ P
    lfp_pop = I_ex + 1.65 * I_in                  # (n, T) per-target net current

    Nv = mp.N.astype(float)
    roi = (lfp_pop * Nv[:, None]).sum(0) / Nv.sum()
    layer = {}
    for lay in mp.layers:
        idx = [i for i, pl in enumerate(mp.pop_layer) if pl == lay]
        w = Nv[idx]
        layer[lay] = (lfp_pop[idx] * w[:, None]).sum(0) / w.sum()

    # full-length z-score using post-warmup statistics (matches PSD convention)
    keep = t_axis > result["warmup"]
    roi_z = (roi - roi[keep].mean()) / roi[keep].std()
    return dict(t=t_axis, roi=roi, roi_z=roi_z, layer=layer, pop=lfp_pop)


def zscore(sig):
    sig = np.asarray(sig, dtype=float)
    return (sig - sig.mean()) / sig.std()


# --------------------------------------------------------------------------- #
#  Power spectral density                                                      #
# --------------------------------------------------------------------------- #
def power_spectrum(sig, fs, method="welch", nperseg=None, nfft_factor=0):
    """One-sided PSD. 'welch' (Hann, averaged) or 'fft' (single-shot).

    For 'welch', the default ``nperseg`` is half the signal so the real
    frequency resolution is fs/(len/2) ~= 2/duration (a few averaging segments
    at 50% overlap). ``nfft = nfft_factor * nperseg`` zero-pads each segment,
    interpolating the spectrum onto a denser, smoother frequency grid without
    altering the underlying resolution.
    """
    sig = np.asarray(sig, dtype=float)
    if method == "welch":
        if nperseg is None:
            nperseg = min(len(sig), max(256, len(sig) // 2))
        nfft = max(nperseg, int(nfft_factor) * nperseg)
        f, p = welch(sig, fs=fs, nperseg=nperseg, nfft=nfft,
                     window="hann", scaling="density")
    else:
        n = len(sig)
        f = np.fft.rfftfreq(n, d=1.0 / fs)
        p = np.abs(np.fft.rfft(sig)) ** 2 / n
    return f, p


def band_powers(f, p, bands, flo=0.5, fhi=200.0):
    """Relative power per band (normalised within [flo, fhi])."""
    sel = (f >= flo) & (f <= fhi)
    tot = p[sel].sum()
    return {b: float(p[(f >= lo) & (f < hi)].sum() / tot) for b, (lo, hi) in bands.items()}


# --------------------------------------------------------------------------- #
#  Morlet wavelet scalogram                                                    #
# --------------------------------------------------------------------------- #
def wavelet_scalogram(sig, fs, fmin=2.0, fmax=80.0, n_scales=64, w=6.0):
    """
    Morlet continuous wavelet transform on log-spaced frequencies.

    Returns (freqs, t_sec, power[n_scales, N]).
    """
    from scipy.signal import fftconvolve

    def _morlet2(M, s, w):
        x = (np.arange(M) - (M - 1) / 2.0) / s
        return (np.pi ** -0.25) * np.sqrt(1.0 / s) * np.exp(1j * w * x) * np.exp(-0.5 * x ** 2)

    sig = np.asarray(sig, dtype=float)
    nn = len(sig)
    freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_scales)
    scales = w * fs / (2.0 * np.pi * freqs)
    coef = np.empty((n_scales, nn), dtype=complex)
    for i, sc in enumerate(scales):
        M = min(nn, int(10 * sc))
        wav = _morlet2(M, sc, w)
        coef[i] = fftconvolve(sig, wav, mode="same")
    t_sec = np.arange(nn) / fs
    return freqs, t_sec, np.abs(coef) ** 2


# --------------------------------------------------------------------------- #
#  Cross-frequency theta-beta PLV                                             #
# --------------------------------------------------------------------------- #
def _band_phase(sig, fs, band):
    """Instantaneous phase of ``sig`` in ``band`` (SOS bandpass + Hilbert)."""
    nyq = fs / 2.0
    low = band[0] / nyq
    high = min(band[1], nyq * 0.99) / nyq
    sos = butter(4, [low, high], btype="band", output="sos")
    return np.angle(hilbert(sosfiltfilt(sos, np.asarray(sig, dtype=float))))


def _nm_plv(phase_slow, phase_fast, n, m):
    """n:m phase-locking value, |<exp(i(n*phi_slow - m*phi_fast))>|."""
    return float(np.abs(np.mean(np.exp(1j * (n * phase_slow - m * phase_fast)))))


def theta_beta_plv_matrix(layer_lfps, fs, theta_band, beta_band, n, m, tmin=0.0, t=None):
    """
    Asymmetric layer x layer n:m PLV: rows = theta-phase signal, cols =
    beta-phase signal. Diagonal = within-layer theta-beta coupling.

    Parameters
    ----------
    layer_lfps : dict {layer: 1d signal}
    Returns (labels, plv_matrix[L, L]).
    """
    labels = list(layer_lfps)
    if t is not None and tmin > 0:
        keep = t >= tmin
        sigs = [np.asarray(layer_lfps[l])[keep] for l in labels]
    else:
        sigs = [np.asarray(layer_lfps[l]) for l in labels]
    ph_t = [_band_phase(s, fs, theta_band) for s in sigs]
    ph_b = [_band_phase(s, fs, beta_band) for s in sigs]
    L = len(labels)
    M = np.empty((L, L))
    for i in range(L):
        for j in range(L):
            M[i, j] = _nm_plv(ph_t[i], ph_b[j], n, m)
    return labels, M


def theta_beta_comodulogram(sig, fs, theta_freqs, beta_freqs, bw, n, m, tmin=0.0, t=None):
    """
    Fine-frequency theta-beta PLV map of one signal with itself: for each
    (theta_f, beta_f) centre, narrowband around +/- bw/2 and take the n:m PLV.

    Returns (theta_freqs, beta_freqs, plv[n_theta, n_beta]).
    """
    sig = np.asarray(sig, dtype=float)
    if t is not None and tmin > 0:
        sig = sig[t >= tmin]
    half = bw / 2.0
    ph_t = {ft: _band_phase(sig, fs, (ft - half, ft + half)) for ft in theta_freqs}
    ph_b = {fb: _band_phase(sig, fs, (fb - half, fb + half)) for fb in beta_freqs}
    out = np.empty((len(theta_freqs), len(beta_freqs)))
    for i, ft in enumerate(theta_freqs):
        for j, fb in enumerate(beta_freqs):
            out[i, j] = _nm_plv(ph_t[ft], ph_b[fb], n, m)
    return np.asarray(theta_freqs, float), np.asarray(beta_freqs, float), out


# --------------------------------------------------------------------------- #
#  Neuron-level firing rates: distribution + time course                       #
# --------------------------------------------------------------------------- #
def _pop_layer_groups(result, mp):
    """{layer: [pop_name, ...]} restricted to populations present in the result."""
    groups = {lay: [] for lay in mp.layers}
    for name, pl in zip(mp.pop_names, mp.pop_layer):
        if name in result["pop_gid"]:
            groups[pl].append(name)
    return {lay: names for lay, names in groups.items() if names}


def neuron_firing_rates(result, mp):
    """
    Time-averaged firing rate (Hz) of every single neuron over the post-warmup
    window -- silent neurons contribute an explicit 0, so each population array
    has length ``n`` (the full population). Feeds the per-neuron rate box plot.

    Returns dict with:
      per_pop   : {pop_name: rates[n]}
      per_layer : {layer: rates[sum n over that layer's pops]}
      all       : rates over every recorded neuron
    """
    s, t = result["senders"], result["times"]
    warm, tsim = result["warmup"], result["t_sim"]
    dur = (tsim - warm) / 1000.0
    post = t > warm
    s_post = s[post]

    per_pop = {}
    for name, (g0, g1, n) in result["pop_gid"].items():
        m = (s_post >= g0) & (s_post <= g1)
        counts = np.bincount(s_post[m] - g0, minlength=n)[:n]
        per_pop[name] = counts.astype(float) / dur

    groups = _pop_layer_groups(result, mp)
    per_layer = {lay: np.concatenate([per_pop[name] for name in names])
                 for lay, names in groups.items()}
    all_rates = np.concatenate([per_pop[name] for name in result["pop_gid"]])
    return dict(per_pop=per_pop, per_layer=per_layer, all=all_rates)


def neuron_rate_series(result, mp, bin_ms=5.0):
    """
    Population-mean single-neuron firing rate (Hz) per time bin, over the full
    ``t_sim`` span -- i.e. the per-population/-layer spike count divided by
    (n_neurons x bin width). Feeds the time-domain firing-rate curve plot.

    Returns (t_centres (ms), {pop: rate}, {layer: rate}, total_rate).
    """
    centres, pop_counts = population_spike_count_series(result, bin_ms)
    bin_s = bin_ms / 1000.0
    pop_rate = {name: pop_counts[name].astype(float) / (n * bin_s)
                for name, (_g0, _g1, n) in result["pop_gid"].items()}

    groups = _pop_layer_groups(result, mp)
    layer_rate = {}
    for lay, names in groups.items():
        tot = sum(pop_counts[name] for name in names)
        n_tot = sum(result["pop_gid"][name][2] for name in names)
        layer_rate[lay] = tot.astype(float) / (n_tot * bin_s)

    tot_counts = sum(pop_counts.values())
    n_all = sum(v[2] for v in result["pop_gid"].values())
    total_rate = tot_counts.astype(float) / (n_all * bin_s)
    return centres, pop_rate, layer_rate, total_rate


# --------------------------------------------------------------------------- #
#  Activity-state classification: synchronous/asynchronous x regular/irregular #
# --------------------------------------------------------------------------- #
def isi_cv(result, mp, min_spikes=3):
    """
    Per-neuron ISI coefficient of variation (std/mean of inter-spike intervals)
    over the post-warmup window. CV ~ 0 is clock-like (regular), CV ~ 1 is
    Poisson-like (irregular). Only neurons with >= ``min_spikes`` spikes (>= 2
    ISIs) contribute.

    Returns dict with:
      per_pop   : {pop_name: cv[...]}   (only qualifying neurons)
      per_layer : {layer: cv[...]}
      all       : cv over every qualifying neuron
      mean      : mean CV over all qualifying neurons (nan if none)
    """
    s, t = result["senders"], result["times"]
    warm = result["warmup"]
    post = t > warm
    s_p, t_p = s[post], t[post]
    order = np.lexsort((t_p, s_p))            # sort by sender, then time
    s_s, t_s = s_p[order], t_p[order]

    gid_cv = {}
    if s_s.size:
        uniq, starts = np.unique(s_s, return_index=True)
        bounds = np.append(starts, s_s.size)
        for k, gid in enumerate(uniq):
            seg = t_s[bounds[k]:bounds[k + 1]]
            if seg.size >= min_spikes:
                isi = np.diff(seg)
                mu = isi.mean()
                if mu > 0:
                    gid_cv[int(gid)] = isi.std() / mu

    per_pop = {}
    for name, (g0, g1, _n) in result["pop_gid"].items():
        vals = [gid_cv[g] for g in range(g0, g1 + 1) if g in gid_cv]
        per_pop[name] = np.asarray(vals, dtype=float)

    groups = _pop_layer_groups(result, mp)
    per_layer = {lay: np.concatenate([per_pop[name] for name in names]) if
                 any(per_pop[name].size for name in names) else np.array([])
                 for lay, names in groups.items()}
    all_cv = np.concatenate([per_pop[name] for name in result["pop_gid"]]) \
        if gid_cv else np.array([])
    mean_cv = float(all_cv.mean()) if all_cv.size else float("nan")
    return dict(per_pop=per_pop, per_layer=per_layer, all=all_cv, mean=mean_cv)


def synchrony_chi(result, mp, bin_ms=3.0, max_neurons_per_pop=500, seed=0):
    """
    Golomb-Rinzel population synchrony index chi in [0, 1] (0 = asynchronous,
    1 = fully synchronous), computed on binned post-warmup spike trains:

        chi = sqrt( Var_t(<s>(t)) / mean_i Var_t(s_i(t)) )

    where s_i(t) is neuron i's binned spike count and <s>(t) the population
    average. Each population is sub-sampled to <= ``max_neurons_per_pop`` neurons
    to keep the neuron x bin matrix small; ``overall`` pools those samples.

    Returns dict: per_pop {name: chi}, overall (chi), bin_ms.
    """
    warm, tsim = result["warmup"], result["t_sim"]
    n_bins = max(1, int(round((tsim - warm) / bin_ms)))
    edges = np.linspace(warm, tsim, n_bins + 1)
    s, t = result["senders"], result["times"]
    post = t > warm
    s_p, t_p = s[post], t[post]
    rng = np.random.default_rng(seed)

    def chi_for(gids):
        gids = np.sort(np.asarray(gids))
        if gids.size == 0:
            return 0.0
        idx = np.searchsorted(gids, s_p)
        in_range = idx < gids.size
        idx_c = np.where(in_range, idx, 0)
        valid = in_range & (gids[idx_c] == s_p)
        neuron_idx = idx[valid]
        bin_idx = np.clip(np.searchsorted(edges, t_p[valid], side="right") - 1,
                          0, n_bins - 1)
        S = np.zeros((gids.size, n_bins))
        np.add.at(S, (neuron_idx, bin_idx), 1.0)
        var_ind = S.var(axis=1).mean()
        if var_ind <= 0:
            return 0.0
        return float(np.sqrt(max(S.mean(axis=0).var(), 0.0) / var_ind))

    per_pop, pooled = {}, []
    for name, (g0, g1, n) in result["pop_gid"].items():
        pool = np.arange(g0, g1 + 1)
        if pool.size > max_neurons_per_pop:
            pool = rng.choice(pool, max_neurons_per_pop, replace=False)
        per_pop[name] = chi_for(pool)
        pooled.append(pool)
    overall = chi_for(np.concatenate(pooled)) if pooled else 0.0
    return dict(per_pop=per_pop, overall=overall, bin_ms=float(bin_ms))


def classify_activity_state(mean_cv, chi, cv_thresh=0.5, chi_thresh=0.15):
    """
    Combine ISI-CV (regularity axis) and chi (synchrony axis) into one of the
    four Brunel network states:

        SR  synchronous-regular      SI  synchronous-irregular
        AR  asynchronous-regular     AI  asynchronous-irregular

    ``mean_cv < cv_thresh`` -> regular; ``chi >= chi_thresh`` -> synchronous.
    Thresholds are heuristic; the raw ``mean_cv``/``chi`` are returned so a
    borderline call can be re-judged. ``mean_cv`` may be nan (too few spikes),
    in which case regularity is reported as ``unknown``.
    """
    if np.isnan(mean_cv):
        regularity, reg_code = "unknown", "?"
    elif mean_cv < cv_thresh:
        regularity, reg_code = "regular", "R"
    else:
        regularity, reg_code = "irregular", "I"
    synchronous = chi >= chi_thresh
    synchrony = "synchronous" if synchronous else "asynchronous"
    state = ("S" if synchronous else "A") + reg_code
    return dict(state=state, synchrony=synchrony, regularity=regularity,
                mean_cv=(None if np.isnan(mean_cv) else float(mean_cv)),
                chi=float(chi), cv_thresh=float(cv_thresh),
                chi_thresh=float(chi_thresh))
