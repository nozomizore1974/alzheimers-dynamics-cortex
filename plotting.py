"""
plotting.py -- figures for one column simulation.

  plot_raster_total      : all spikes, one colour, + one bar strip per layer
                           and a total spike-count bar strip below
  plot_raster_subpop     : spikes coloured by sub-population (8 pops x
                           healthy / Abeta-hyper / Abeta-supp / Abeta-I),
                           + total, healthy-only and Abeta-only spike-count
                           bar strips below. Skip this figure when the run has
                           no Abeta subsets (abeta_ratio == 0) -- see main.py.
  plot_lfp               : ROI + per-layer LFP proxy time series
  plot_psd               : power spectral density with band shading
  plot_wavelet           : Morlet scalogram (time x frequency)
  plot_plv_matrix        : layer x layer theta-beta n:m PLV heat-map
  plot_comodulogram      : fine-frequency theta-beta PLV map
  plot_neuron_rate_box   : per-neuron time-averaged firing-rate distribution
                           (box plot), grouped by population and by layer
  plot_neuron_rate_series: population/layer mean single-neuron firing rate as a
                           time-domain curve

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


def _sample_population_neurons(result, neuron_sample_rate):
    """Select a reproducible fraction of every population's neuron GIDs."""
    rate = float(neuron_sample_rate)
    if not 0.0 < rate <= 1.0:
        raise ValueError("neuron_sample_rate must be in the interval (0, 1]")

    rng = np.random.default_rng(0)
    samples = {}
    for name, (g0, g1, n_neurons) in result["pop_gid"].items():
        n_sample = max(1, int(np.ceil(n_neurons * rate)))
        samples[name] = np.sort(
            rng.choice(np.arange(g0, g1 + 1), size=n_sample, replace=False)
        )
    return samples


def _count_strip(ax, t_centres, counts, start_ms, bin_ms, color="0.25", alpha=1.0,
                 hatch=None, label=None, ylabel=None):
    """One spike-count bar series (shared style for total/per-layer/per-population strips)."""
    cm = t_centres >= start_ms
    ax.bar(t_centres[cm], counts[cm], width=bin_ms, color=color, alpha=alpha,
          hatch=hatch, edgecolor=color if hatch else None, linewidth=0.4,
          align="center", label=label)
    ax.set_ylabel(ylabel if ylabel is not None else f"spikes / {bin_ms:g} ms")
    ax.grid(True, axis="y", alpha=0.3)


def plot_raster_total(result, spike_count, layer_spike_count, path, raster_start=None,
                      neuron_sample_rate=1.0):
    """
    Parameters
    ----------
    spike_count : (t_centres_ms, counts) from ``analysis.spike_count_series``.
    layer_spike_count : (t_centres_ms, {layer: counts}) from
        ``analysis.layer_spike_count_series``.
    raster_start : float, optional
        First displayed time in ms. Defaults to the simulation warmup time.
    neuron_sample_rate : float, optional
        Fraction of neurons sampled independently within every population.

    Below the raster: one bar strip per layer (own panel each), then the total
    spike-count bar strip at the bottom (shared time axis throughout).
    """
    s, t = result["senders"], result["times"]
    start_ms = result["warmup"] if raster_start is None else float(raster_start)
    samples = _sample_population_neurons(result, neuron_sample_rate)
    sampled_gids = np.sort(np.concatenate(list(samples.values())))
    m = (t >= start_ms) & np.isin(s, sampled_gids)
    ss, tt = s[m], t[m]
    sampled_rows = np.searchsorted(sampled_gids, ss)

    e_mask = np.zeros(ss.size, dtype=bool)
    i_mask = np.zeros(ss.size, dtype=bool)
    for sub in result["subsets"]:
        in_range = (ss >= sub["g0"]) & (ss <= sub["g1"])
        if sub["pop"].endswith("E"):
            e_mask |= in_range
        else:
            i_mask |= in_range

    lc_t, lc_counts = layer_spike_count
    layers = list(lc_counts)
    n_layers = len(layers)
    height_ratios = [4] + [1] * n_layers + [1]
    fig, axes = plt.subplots(len(height_ratios), 1, figsize=(9, 7 + 2 * len(height_ratios)),
                             sharex=True,
                             gridspec_kw={"height_ratios": height_ratios, "hspace": 0.08})
    ax = axes[0]

    ax.plot(tt[e_mask], sampled_rows[e_mask], ".", ms=1.0, color="#1f77b4",
            rasterized=True, label="E")
    ax.plot(tt[i_mask], sampled_rows[i_mask], ".", ms=1.0, color="#d62728",
            rasterized=True, label="I")

    # y-axis ticks = each population centred on its (contiguous) GID band, labelled
    # by layer + E/I (e.g. 'II/III E'); faint lines mark population boundaries.
    yticks, ylabels = [], []
    y0 = 0
    for name in result["pop_gid"]:
        n_sample = len(samples[name])
        lo, hi = y0, y0 + n_sample - 1
        yticks.append((lo + hi) / 2); ylabels.append(_pop_label(name))
        ax.axhline(hi + 0.5, color="0.8", lw=0.5)
        y0 += n_sample
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels); ax.invert_yaxis()
    ax.set_ylabel("layer / population")
    ax.set_xlim(start_ms, result["t_sim"]); ax.set_title("Total raster")
    ax.legend(markerscale=6, fontsize=8, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.setp(ax.get_xticklabels(), visible=False)

    sc_t, sc_counts = spike_count
    bin_ms = float(sc_t[1] - sc_t[0]) if len(sc_t) > 1 else 1.0

    for ax_l, lay in zip(axes[1:1 + n_layers], layers):
        _count_strip(ax_l, lc_t, lc_counts[lay], start_ms, bin_ms,
                    color=_LAYER_COLOR.get(lay), ylabel=lay)
        plt.setp(ax_l.get_xticklabels(), visible=False)

    axc = axes[-1]
    _count_strip(axc, sc_t, sc_counts, start_ms, bin_ms, ylabel=f"total\n(spikes/{bin_ms:g}ms)")
    axc.set_xlabel("time (ms)")

    fig.savefig(path, dpi=600, bbox_inches="tight"); plt.close(fig)
    return path


def plot_raster_subpop(result, mp, spike_count, health_spike_count, path, raster_start=None,
                       neuron_sample_rate=1.0):
    """
    Parameters
    ----------
    spike_count : (t_centres_ms, counts) from ``analysis.spike_count_series``,
        drawn as the total spike-count bar strip below the raster.
    health_spike_count : (t_centres_ms, {"healthy": counts, "abeta": counts})
        from ``analysis.health_spike_count_series``, drawn as two further bar
        strips below that: healthy subsets, then Abeta-affected subsets.
    raster_start : float, optional
        First displayed time in ms. Defaults to the simulation warmup time.
    neuron_sample_rate : float, optional
        Fraction of neurons sampled independently within every population.
    """
    s, t = result["senders"], result["times"]
    start_ms = result["warmup"] if raster_start is None else float(raster_start)
    samples = _sample_population_neurons(result, neuron_sample_rate)
    fig, (ax, axc, axh, axa) = plt.subplots(4, 1, figsize=(11, 16), sharex=True,
                                            gridspec_kw={"height_ratios": [4, 1, 1, 1], "hspace": 0.08})
    y0, yticks, ylabels = 0, [], []
    seen = set()
    for name in mp.pop_names:
        block0 = y0
        pop_sample = samples[name]
        for r in [x for x in result["subsets"] if x["pop"] == name]:
            subset_sample = pop_sample[(pop_sample >= r["g0"]) & (pop_sample <= r["g1"])]
            m = (t >= start_ms) & np.isin(s, subset_sample)
            tt = t[m]
            ss = np.searchsorted(subset_sample, s[m]) + y0
            col = _subset_color(name, r["kind"])
            lbl = r["kind"] if r["kind"] not in seen else None
            seen.add(r["kind"])
            ax.plot(tt, ss, ".", ms=1.0, color=col, rasterized=True, label=lbl)
            y0 += len(subset_sample)
        yticks.append((block0 + y0) / 2); ylabels.append(name)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels); ax.invert_yaxis()
    ax.set_xlim(start_ms, result["t_sim"])
    ax.set_title("Raster by sub-population (colour = layer; shade = healthy/Abeta subset)")
    ax.legend(markerscale=6, fontsize=8, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0)
    plt.setp(ax.get_xticklabels(), visible=False)

    sc_t, sc_counts = spike_count
    bin_ms = float(sc_t[1] - sc_t[0]) if len(sc_t) > 1 else 1.0
    _count_strip(axc, sc_t, sc_counts, start_ms, bin_ms, ylabel=f"total\n(spikes/{bin_ms:g}ms)")
    plt.setp(axc.get_xticklabels(), visible=False)

    hc_t, hc_counts = health_spike_count
    _count_strip(axh, hc_t, hc_counts["healthy"], start_ms, bin_ms, color="#1f77b4",
                ylabel=f"healthy\n(spikes/{bin_ms:g}ms)")
    plt.setp(axh.get_xticklabels(), visible=False)

    _count_strip(axa, hc_t, hc_counts["abeta"], start_ms, bin_ms, color="#d62728",
                ylabel=f"abeta\n(spikes/{bin_ms:g}ms)")
    axa.set_xlabel("time (ms)")

    fig.savefig(path, dpi=600, bbox_inches="tight"); plt.close(fig)
    return path


def plot_lfp(lfp, warmup, path):
    t, roi, roi_z, layer = lfp["t"], lfp["roi"], lfp["roi_z"], lfp["layer"]
    m = t > warmup
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    axes[0].plot(t[m], roi[m], lw=0.7, color="k")
    axes[0].set_ylabel("ROI LFP (a.u.)"); axes[0].set_title("LFP proxy")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t[m], roi_z[m], lw=0.7, color="C3")
    axes[1].set_ylabel("ROI LFP (z)"); axes[1].grid(True, alpha=0.3)
    off = 0.0
    for lay, sig in layer.items():
        z = (sig[m] - sig[m].mean()) / sig[m].std()
        axes[2].plot(t[m], z + off, lw=0.6, color=_LAYER_COLOR.get(lay), label=lay); off += 6
    axes[2].set_yticks([]); axes[2].set_xlabel("time (ms)")
    axes[2].set_ylabel("per-layer (z, offset)"); axes[2].legend(ncol=4, fontsize=8)
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=600); plt.close(fig)
    return path


def plot_psd(f, p, bands, path, fmax=80.0):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(f, p, color="k", lw=1.2)
    cols = plt.cm.tab10(np.linspace(0, 1, len(bands)))
    for (b, (lo, hi)), c in zip(bands.items(), cols):
        ax.axvspan(lo, hi, color=c, alpha=0.13, label=b)
    ax.set_xlim(0, fmax); ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("PSD")
    ax.set_title("LFP power spectrum"); ax.legend(fontsize=8, ncol=len(bands))
    fig.tight_layout(); fig.savefig(path, dpi=600); plt.close(fig)
    return path


def plot_wavelet(freqs, t_sec, power, warmup, path):
    m = t_sec > (warmup / 1000.0)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    pm = ax.pcolormesh(t_sec[m] * 1000.0, freqs, np.log10(power[:, m] + 1e-20),
                       shading="auto", cmap="magma")
    ax.set_yscale("log"); ax.set_ylabel("frequency (Hz)"); ax.set_xlabel("time (ms)")
    ax.set_title("Morlet wavelet scalogram (log10 power)")
    fig.colorbar(pm, ax=ax, label="log10 power")
    fig.tight_layout(); fig.savefig(path, dpi=600); plt.close(fig)
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
    fig.tight_layout(); fig.savefig(path, dpi=600); plt.close(fig)
    return path


def plot_comodulogram(theta_f, beta_f, plv, n, m, path):
    fig, ax = plt.subplots(figsize=(6, 4.6))
    pm = ax.pcolormesh(beta_f, theta_f, plv, vmin=0, vmax=1, shading="auto", cmap="inferno")
    ax.set_xlabel(r"$\beta$ sub-frequency (Hz)"); ax.set_ylabel(r"$\theta$ sub-frequency (Hz)")
    ax.set_title(f"theta-beta PLV comodulogram (ROI LFP, n:m = {n}:{m})")
    fig.colorbar(pm, ax=ax, label="PLV")
    fig.tight_layout(); fig.savefig(path, dpi=600); plt.close(fig)
    return path


def _rate_boxplot(ax, labels, data, colors, title):
    """Shared box-plot styling for a per-neuron firing-rate distribution panel."""
    bp = ax.boxplot(data, showfliers=False, patch_artist=True, whis=(5, 95))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    for med in bp["medians"]:
        med.set_color("k")
    means = [float(d.mean()) if len(d) else 0.0 for d in data]
    ax.plot(range(1, len(data) + 1), means, "D", ms=4, color="k", label="mean")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("firing rate (Hz)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")


def plot_neuron_rate_box(neuron_rates, path, state=None):
    """
    Box plot of every neuron's full-window time-averaged firing rate, one box
    per population (left) and per layer (right). Whiskers span the 5-95th
    percentiles, black diamond = mean. ``state`` (optional dict from
    ``analysis.classify_activity_state``) is annotated in the figure title.

    Parameters
    ----------
    neuron_rates : dict from ``analysis.neuron_firing_rates`` (per_pop, per_layer).
    """
    per_pop, per_layer = neuron_rates["per_pop"], neuron_rates["per_layer"]
    pop_names = list(per_pop)
    layer_names = list(per_layer)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 5.2),
        gridspec_kw={"width_ratios": [max(len(pop_names), 1), max(len(layer_names), 1)]})

    _rate_boxplot(ax1, [_pop_label(n) for n in pop_names],
                  [per_pop[n] for n in pop_names],
                  [_LAYER_COLOR.get(n[:-1], "#888888") for n in pop_names],
                  "Per-neuron time-averaged rate by population")
    _rate_boxplot(ax2, layer_names,
                  [per_layer[l] for l in layer_names],
                  [_LAYER_COLOR.get(l, "#888888") for l in layer_names],
                  "by layer")

    if state:
        tag = f"state: {state['state']} ({state['synchrony']} / {state['regularity']})"
        fig.suptitle(tag, fontsize=11, y=1.02)
    fig.tight_layout(); fig.savefig(path, dpi=600, bbox_inches="tight"); plt.close(fig)
    return path


def plot_neuron_rate_series(centres, pop_rate, layer_rate, total_rate, path,
                            start_ms=0.0):
    """
    Time-domain firing-rate curve: population-mean single-neuron rate (Hz) per
    time bin. Top panel = the 8 populations (E solid, I dashed, coloured by
    layer); bottom = per-layer means plus the whole-column total.

    Parameters
    ----------
    centres, pop_rate, layer_rate, total_rate : from ``analysis.neuron_rate_series``.
    start_ms : float
        First displayed time in ms (mirrors the rasters' ``raster_start``).
    """
    centres = np.asarray(centres)
    cm = centres >= start_ms
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                   gridspec_kw={"hspace": 0.12})

    for name, r in pop_rate.items():
        ax1.plot(centres[cm], np.asarray(r)[cm], lw=0.7,
                 color=_LAYER_COLOR.get(name[:-1], "#888888"),
                 ls="-" if name.endswith("E") else "--", label=_pop_label(name))
    ax1.set_ylabel("pop mean rate (Hz)")
    ax1.set_title("Population mean single-neuron firing rate over time")
    ax1.legend(ncol=1, fontsize=7, loc="upper left",
               bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.get_xticklabels(), visible=False)

    for lay, r in layer_rate.items():
        ax2.plot(centres[cm], np.asarray(r)[cm], lw=0.9,
                 color=_LAYER_COLOR.get(lay, "#888888"), label=lay)
    ax2.plot(centres[cm], np.asarray(total_rate)[cm], lw=1.1, color="k", label="total")
    ax2.set_ylabel("layer mean rate (Hz)"); ax2.set_xlabel("time (ms)")
    ax2.legend(ncol=1, fontsize=8, loc="upper left",
               bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax2.grid(True, alpha=0.3)
    if centres.size:
        ax2.set_xlim(start_ms, centres[-1])

    fig.savefig(path, dpi=600, bbox_inches="tight"); plt.close(fig)
    return path
