"""
plotting.py -- figures for one column simulation.

  plot_raster_total      : all spikes, one colour
  plot_raster_subpop     : spikes coloured by sub-population (8 pops x
                           healthy / Abeta-hyper / Abeta-supp / Abeta-I)
  plot_lfp               : ROI + per-layer LFP proxy time series
  plot_psd               : power spectral density with band shading
  plot_wavelet           : Morlet scalogram (time x frequency)
  plot_plv_matrix        : layer x layer theta-beta n:m PLV heat-map
  plot_comodulogram      : fine-frequency theta-beta PLV map

Self-contained: matplotlib + numpy. Uses a non-interactive backend so it runs
headless. Labels are ASCII so no CJK font is needed.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Each (E/I × health-status) combination gets a unique hue.
# E group: cool hues (blue → cyan → violet → orchid)
# I group: warm hues (orange → yellow-green → red → brown)
_SUBSET_COLOR = {
    ("E", "healthy"):     "#1f77b4",  # blue
    ("E", "abeta_supp"):  "#17becf",  # cyan
    ("E", "abeta"):       "#9467bd",  # violet
    ("E", "abeta_hyper"): "#e377c2",  # orchid
    ("I", "healthy"):     "#ff7f0e",  # orange
    ("I", "abeta_supp"):  "#bcbd22",  # yellow-green
    ("I", "abeta"):       "#d62728",  # red
    ("I", "abeta_hyper"): "#8c564b",  # brown
}
_LAYER_COLOR = {"L23": "#5ba3d9", "L4": "#74c476", "L5": "#e6550d", "L6": "#9467bd"}

# Cortical layer label -> Roman-numeral convention used on figures.
_LAYER_ROMAN = {"1": "I", "23": "II/III", "2": "II", "3": "III",
                "4": "IV", "5": "V", "6": "VI"}


def _pop_label(name):
    """'L23E' -> 'II/III E', 'L4I' -> 'IV I' (layer Roman numeral + E/I)."""
    layer = name[:-1].lstrip("L")
    return f"{_LAYER_ROMAN.get(layer, layer)} {name[-1]}"


def _subset_color(pop, kind):
    ei = pop[-1]  # "E" or "I"
    return _SUBSET_COLOR.get((ei, kind), "#888888")


def _sample(x, y, nmax=6000, rng=None):
    if x.size > nmax:
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(x.size, nmax, replace=False)
        return x[idx], y[idx]
    return x, y


def plot_raster_total(result, path, nmax=12000):
    s, t = result["senders"], result["times"]
    warm = result["warmup"]
    g_lo = min(v[0] for v in result["pop_gid"].values())
    m = t > warm
    ss, tt = s[m], t[m]

    e_mask = np.zeros(ss.size, dtype=bool)
    i_mask = np.zeros(ss.size, dtype=bool)
    for sub in result["subsets"]:
        in_range = (ss >= sub["g0"]) & (ss <= sub["g1"])
        if sub["pop"].endswith("E"):
            e_mask |= in_range
        else:
            i_mask |= in_range

    rng = np.random.default_rng(0)
    nmax_ei = nmax // 2
    fig, ax = plt.subplots(figsize=(10, 8))
    te, se = _sample(tt[e_mask], (ss[e_mask] - g_lo), nmax_ei, rng)
    ax.plot(te, se, ".", ms=1.0, color="#1f77b4", rasterized=True, label="E")
    ti, si = _sample(tt[i_mask], (ss[i_mask] - g_lo), nmax_ei, rng)
    ax.plot(ti, si, ".", ms=1.0, color="#d62728", rasterized=True, label="I")

    # y-axis ticks = each population centred on its (contiguous) GID band, labelled
    # by layer + E/I (e.g. 'II/III E'); faint lines mark population boundaries.
    yticks, ylabels = [], []
    for name, (g0, g1, _n) in result["pop_gid"].items():
        lo, hi = g0 - g_lo, g1 - g_lo
        yticks.append((lo + hi) / 2); ylabels.append(_pop_label(name))
        ax.axhline(hi + 0.5, color="0.8", lw=0.5)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels); ax.invert_yaxis()
    ax.set_xlabel("time (ms)"); ax.set_ylabel("layer / population")
    ax.set_xlim(warm, result["t_sim"]); ax.set_title("Total raster")
    ax.legend(markerscale=6, fontsize=8, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0)
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return path


def plot_raster_subpop(result, mp, path, nmax_per=3000):
    s, t = result["senders"], result["times"]
    warm = result["warmup"]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(11, 9))
    y0, yticks, ylabels = 0, [], []
    seen = set()
    for name in mp.pop_names:
        block0 = y0
        for r in [x for x in result["subsets"] if x["pop"] == name]:
            m = (s >= r["g0"]) & (s <= r["g1"]) & (t > warm)
            tt, ss = _sample(t[m], (s[m] - r["g0"] + y0), nmax_per, rng)
            col = _subset_color(name, r["kind"])
            lbl = r["kind"] if r["kind"] not in seen else None
            seen.add(r["kind"])
            ax.plot(tt, ss, ".", ms=1.0, color=col, rasterized=True, label=lbl)
            y0 += r["n"]
        yticks.append((block0 + y0) / 2); ylabels.append(name)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels); ax.invert_yaxis()
    ax.set_xlim(warm, result["t_sim"]); ax.set_xlabel("time (ms)")
    ax.set_title("Raster by sub-population (colour = layer; shade = healthy/Abeta subset)")
    ax.legend(markerscale=6, fontsize=8, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0)
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    return path


def plot_lfp(lfp, warmup, path):
    t, roi, layer = lfp["t"], lfp["roi"], lfp["layer"]
    m = t > warmup
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(t[m], roi[m], lw=0.7, color="k")
    axes[0].set_ylabel("ROI LFP (a.u.)"); axes[0].set_title("LFP proxy")
    axes[0].grid(True, alpha=0.3)
    off = 0.0
    for lay, sig in layer.items():
        z = (sig[m] - sig[m].mean()) / sig[m].std()
        axes[1].plot(t[m], z + off, lw=0.6, color=_LAYER_COLOR.get(lay), label=lay); off += 6
    axes[1].set_yticks([]); axes[1].set_xlabel("time (ms)")
    axes[1].set_ylabel("per-layer (z, offset)"); axes[1].legend(ncol=4, fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def plot_psd(f, p, bands, path, fmax=80.0):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(f, p, color="k", lw=1.2)
    cols = plt.cm.tab10(np.linspace(0, 1, len(bands)))
    for (b, (lo, hi)), c in zip(bands.items(), cols):
        ax.axvspan(lo, hi, color=c, alpha=0.13, label=b)
    ax.set_xlim(0, fmax); ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("PSD")
    ax.set_title("LFP power spectrum"); ax.legend(fontsize=8, ncol=len(bands))
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def plot_wavelet(freqs, t_sec, power, warmup, path):
    m = t_sec > (warmup / 1000.0)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    pm = ax.pcolormesh(t_sec[m] * 1000.0, freqs, np.log10(power[:, m] + 1e-20),
                       shading="auto", cmap="magma")
    ax.set_yscale("log"); ax.set_ylabel("frequency (Hz)"); ax.set_xlabel("time (ms)")
    ax.set_title("Morlet wavelet scalogram (log10 power)")
    fig.colorbar(pm, ax=ax, label="log10 power")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def plot_plv_matrix(labels, M, n, m, path):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel(r"$\beta$-phase layer"); ax.set_ylabel(r"$\theta$-phase layer")
    ax.set_title(f"theta-beta PLV (n:m = {n}:{m})")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="w" if M[i, j] < 0.6 else "k", fontsize=8)
    fig.colorbar(im, ax=ax, label="PLV")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def plot_comodulogram(theta_f, beta_f, plv, n, m, path):
    fig, ax = plt.subplots(figsize=(6, 4.6))
    pm = ax.pcolormesh(beta_f, theta_f, plv, vmin=0, vmax=1, shading="auto", cmap="inferno")
    ax.set_xlabel(r"$\beta$ sub-frequency (Hz)"); ax.set_ylabel(r"$\theta$ sub-frequency (Hz)")
    ax.set_title(f"theta-beta PLV comodulogram (ROI LFP, n:m = {n}:{m})")
    fig.colorbar(pm, ax=ax, label="PLV")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path
