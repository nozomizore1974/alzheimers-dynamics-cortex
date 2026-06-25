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

# distinct colour per (pop, kind) sub-population
_KIND_SHADE = {"healthy": 1.0, "abeta": 0.55, "abeta_hyper": 0.45, "abeta_supp": 0.75}
_POP_COLOR = {"L23": "#1f77b4", "L4": "#2ca02c", "L5": "#d62728", "L6": "#9467bd"}


def _subset_color(pop, kind):
    base = np.array(matplotlib.colors.to_rgb(_POP_COLOR[pop[:-1]]))
    shade = _KIND_SHADE.get(kind, 1.0)
    return tuple(np.clip(base * shade + (1 - shade) * 0.15, 0, 1))


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
    tt, ss = _sample(t[m], (s[m] - g_lo), nmax)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(tt, ss, ".", ms=1.0, color="k", rasterized=True)
    ax.set_xlabel("time (ms)"); ax.set_ylabel("neuron #")
    ax.set_xlim(warm, result["t_sim"]); ax.set_title("Total raster")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def plot_raster_subpop(result, mp, path, nmax_per=3000):
    s, t = result["senders"], result["times"]
    warm = result["warmup"]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(11, 6))
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
    ax.legend(markerscale=6, fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path


def plot_lfp(lfp, warmup, path):
    t, roi, layer = lfp["t"], lfp["roi"], lfp["layer"]
    m = t > warmup
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(t[m], roi[m], lw=0.7, color="k")
    axes[0].set_ylabel("ROI LFP (a.u.)"); axes[0].set_title("LFP proxy")
    off = 0.0
    for lay, sig in layer.items():
        z = (sig[m] - sig[m].mean()) / sig[m].std()
        axes[1].plot(t[m], z + off, lw=0.6, label=lay); off += 6
    axes[1].set_yticks([]); axes[1].set_xlabel("time (ms)")
    axes[1].set_ylabel("per-layer (z, offset)"); axes[1].legend(ncol=4, fontsize=8)
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
