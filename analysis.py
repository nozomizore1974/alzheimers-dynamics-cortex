"""
analysis.py -- results analysis for one column simulation.

Provides: firing rates, kernel LFP proxy (ROI + per layer), power spectral
density (Welch / FFT), Morlet wavelet scalogram, and cross-frequency theta-beta
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
