"""
main.py -- end-to-end pipeline for one cortical-column simulation.

Flow:  read data  ->  compute model parameters  ->  set up & run simulator
       ->  analyse results  ->  draw figures.

Run:  python main.py [path/to/config.json]
Outputs go to ``<output_dir>/<exp>/`` (figures + a small metrics.json), where
``output_dir`` is the config's top-level ``output_dir`` field (default "out",
relative to the current working directory unless given as an absolute path).

Re-runs are cheap: the NEST simulation and the heavy analysis transforms are
memoised under ``out/<exp>/cache/`` (see ``cache.py``), keyed by a hash of the
parameters that produced them.  Changing a relevant config value invalidates
just the affected entries; to force a full recompute delete that cache folder.

Fully self-contained: imports only this package's modules + numpy/scipy/
matplotlib/nest. No dependency on any external project.
"""
import os
import sys
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import analysis as AN
import model_params as MP
import plotting as PL
import logger as _logger
import cache
import data_loader
import simulator

import numpy as np


def main(config_path=None, n_threads=None):
    """
    Parameters
    ----------
    n_threads : int, optional
        Overrides ``cfg["simulation"]["n_threads"]`` (NEST kernel
        ``local_num_threads``). Lets the caller (e.g. Snakemake's ``threads``
        directive) pick the thread count independently of the config file.
    """
    t0 = time.time()
    log = _logger.get_logger("main")

    # ---------------- 1. read data ----------------
    cfg = data_loader.load_config(config_path) if config_path else data_loader.load_config()
    if n_threads is not None:
        cfg["simulation"]["n_threads"] = n_threads
    mc = data_loader.load_microcircuit()
    nps = data_loader.load_neuron_params()
    exp = cfg["name"]
    syn = cfg["synapse"]
    spec = data_loader.build_spec(nps, cfg["model"],
                              tau_syn_ex=syn.get("tau_syn_ex"),
                              tau_syn_in=syn.get("tau_syn_in"))
    log.info(f"[spec] -- {exp} --")
    outpath = os.path.join(os.getcwd(), cfg.get("output_dir", "out"), exp)
    os.makedirs(outpath, exist_ok=True)
    # Persist the full resolved config so the run can be re-analysed later
    # (e.g. by dynamics.load_run) with the exact parameters that produced it.
    with open(os.path.join(outpath, "config_used.json"), "w") as fp:
        json.dump(cfg, fp, indent=2)
    log.info(f"[data] model={cfg['model']} abeta_ratio={cfg['abeta_ratio']} "
             f"N_scale={cfg['N_scale']} pops={len(mc['pop_names'])}")

    # ---------------- 2. compute model parameters ----------------
    mp = MP.compute_model_params(mc, spec, cfg)
    log.info(f"[params] neurons/col={int(mp.N.sum())} internal synapses={int(mp.n_syn.sum()):,} "
             f"kfac={cfg['weight_ref_scale']/cfg['N_scale']:.3f}")

    # Hash of everything that determines the spike output: the non-analysis
    # config plus a fingerprint of the derived network. Analysis results are
    # keyed off this, so re-simulating is only triggered by a relevant change.
    sim_key = cache.param_hash({k: v for k, v in cfg.items() if k != "analysis"},
                               int(mp.N.sum()), int(mp.n_syn.sum()))
    cdir = os.path.join(outpath, "cache")

    # ---------------- 3. simulate (memoised) ----------------
    result = cache.cached(os.path.join(cdir, "sim.pkl"), sim_key,
                          lambda: simulator.build_and_simulate(mp, cfg, logger=log.info,
                                                               data_path=outpath),
                          logger=log.info)

    # Population GID ranges: cached as their own small object (so downstream
    # tools can look them up without unpickling the full spike train) and
    # written out as a human-readable txt alongside the figures.
    pop_gid = cache.cached(os.path.join(cdir, "pop_gid.pkl"), sim_key,
                           lambda: result["pop_gid"], logger=log.info)
    with open(os.path.join(outpath, "pop_gid.txt"), "w") as fp:
        fp.write(f"{'population':<10}{'g0':>10}{'g1':>10}{'n':>10}\n")
        for name, (g0, g1, n) in pop_gid.items():
            fp.write(f"{name:<10}{g0:>10}{g1:>10}{n:>10}\n")

    # ---------------- 4. analyse (each heavy product memoised) ----------------
    res_dt = cfg["simulation"]["resolution"]
    fs = 1000.0 / res_dt
    warm = cfg["simulation"]["warmup"]
    ana = cfg["analysis"]
    raster_start = float(ana.get("raster_start", warm))
    if not 0.0 <= raster_start < cfg["simulation"]["t_sim"]:
        raise ValueError(
            "analysis.raster_start must be at least 0 and smaller than simulation.t_sim"
        )
    raster_neuron_sample_rate = float(ana.get("raster_neuron_sample_rate", 1.0))
    if not 0.0 < raster_neuron_sample_rate <= 1.0:
        raise ValueError(
            "analysis.raster_neuron_sample_rate must be in the interval (0, 1]"
        )

    def memo(name, compute, *key_parts):
        """Compute-or-load one analysis product, keyed by sim + its params."""
        return cache.cached(os.path.join(cdir, name + ".pkl"),
                            cache.param_hash(sim_key, *key_parts), compute,
                            logger=log.info)

    pop_rates, subset_rates = memo("rates", lambda: AN.firing_rates(result))
    sc_bin = ana.get("spike_count_bin", 1.0)
    spike_count = memo("spike_count", lambda: AN.spike_count_series(result, sc_bin), sc_bin)
    layer_spike_count = memo("layer_spike_count",
        lambda: AN.layer_spike_count_series(result, mp, sc_bin), sc_bin)
    health_spike_count = memo("health_spike_count",
        lambda: AN.health_spike_count_series(result, sc_bin), sc_bin)
    lfp = memo("lfp", lambda: AN.lfp_proxy(result, mp, res_dt), res_dt, "roi_z")

    roi_z = lfp["roi_z"][lfp["t"] > warm]
    f, p = memo("psd", lambda: AN.power_spectrum(roi_z, fs, method=ana["psd_method"]),
                ana["psd_method"], warm)
    bands = {b: tuple(v) for b, v in ana["bands"].items()}
    bp = AN.band_powers(f, p, bands)

    wv = ana["wavelet"]
    wf, wt, wp = memo("wavelet",
        lambda: AN.wavelet_scalogram(lfp["roi_z"], fs, wv["fmin"], wv["fmax"],
                                     wv["n_scales"], wv["w"]),
        wv, "roi_z")

    plv = ana["plv"]
    nm = plv["nm"]
    labels, plv_M = memo("plv_matrix",
        lambda: AN.theta_beta_plv_matrix(
            lfp["layer"], fs, tuple(plv["theta_band"]), tuple(plv["beta_band"]),
            nm[0], nm[1], tmin=warm, t=lfp["t"]),
        plv["theta_band"], plv["beta_band"], nm, warm)
    th_f, be_f, plv_C = memo("comodulogram",
        lambda: AN.theta_beta_comodulogram(
            lfp["roi"], fs, plv["theta_subfreqs"], plv["beta_subfreqs"], plv["sub_bw"],
            nm[0], nm[1], tmin=warm, t=lfp["t"]),
        plv["theta_subfreqs"], plv["beta_subfreqs"], plv["sub_bw"], nm, warm)

    log.info("[rates] " + " ".join(f"{k}:{v:.1f}" for k, v in pop_rates.items()))
    log.info("[bands] " + " ".join(f"{b}:{bp[b]:.2f}" for b in bands))
    log.info(f"[plv] theta-beta {nm[0]}:{nm[1]} diag(mean)={np.mean(np.diag(plv_M)):.3f} "
             f"comodulogram max={plv_C.max():.3f}")

    # ---------------- 5. plot ----------------
    figs = {
        "raster_total":   PL.plot_raster_total(result, spike_count, layer_spike_count,
                                               os.path.join(outpath, "raster_total.png"),
                                               raster_start=raster_start,
                                               neuron_sample_rate=raster_neuron_sample_rate),
    }
    # A healthy-vs-Abeta breakdown is meaningless when the run has no Abeta
    # subsets at all (abeta_ratio == 0 -> simulator never creates them).
    if cfg["abeta_ratio"] > 0:
        figs["raster_subpop"] = PL.plot_raster_subpop(result, mp, spike_count, health_spike_count,
                                                       os.path.join(outpath, "raster_subpop.png"),
                                                       raster_start=raster_start,
                                                       neuron_sample_rate=raster_neuron_sample_rate)
    figs.update({
        "lfp":            PL.plot_lfp(lfp, warm, os.path.join(outpath, "lfp.png")),
        "psd":            PL.plot_psd(f, p, bands, os.path.join(outpath, "psd.png"), fmax=wv["fmax"]),
        "wavelet":        PL.plot_wavelet(wf, wt, wp, warm, os.path.join(outpath, "wavelet.png")),
        "plv_matrix":     PL.plot_plv_matrix(labels, plv_M, nm[0], nm[1],
                                             os.path.join(outpath, "plv_theta_beta_matrix.png")),
        "plv_comodulogram": PL.plot_comodulogram(th_f, be_f, plv_C, nm[0], nm[1],
                                                 os.path.join(outpath, "plv_theta_beta_comodulogram.png")),
    })

    # ---------------- 5b. dynamics: structural + mean-field stability ----------------
    # Predicts, from the wiring + neuron parameters, whether the async-irregular
    # state is stable or the column is driven to a synchronous-regular one, and
    # which populations / frequency drive it. Guarded: never breaks the pipeline.
    dyn_metrics = None
    try:
        import dynamics as DYN
        dyn = DYN.analyze(mp, pop_rates, outpath)
        dyn_metrics = {"stability": dyn["stability"],
                       "spectral_radius": dyn["structural"]["spectral_radius"]}
        log.info(f"[dynamics] margin={dyn['stability']['margin']:.2f} "
                 f"f*={dyn['stability']['critical_freq']:.0f}Hz "
                 f"rho={dyn['structural']['spectral_radius']:.2g}")
    except Exception as err:  # noqa: BLE001
        log.info(f"[dynamics] skipped ({err})")

    metrics = {
        "config": {k: cfg[k] for k in ("model", "abeta_ratio", "N_scale")},
        "neurons": int(mp.N.sum()), "internal_synapses": int(mp.n_syn.sum()),
        "pop_rates": pop_rates, "band_powers": bp,
        "plv_nm": nm, "plv_diag_mean": float(np.mean(np.diag(plv_M))),
        "dynamics": dyn_metrics,
        "figures": {k: os.path.relpath(v, HERE) for k, v in figs.items()},
    }
    with open(os.path.join(outpath, "metrics.json"), "w") as fp:
        json.dump(metrics, fp, indent=2)

    log.info(f"[done] {time.time()-t0:.0f}s -> {len(figs)} figures + metrics.json in {os.path.relpath(outpath, HERE)}/")
    return metrics


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", nargs="?", default="data/aeif_baseline_template.yaml",
                   help="path to config YAML/JSON (default: data/config.yaml)")
    p.add_argument("--threads", "-j", type=int, default=None,
                   help="override simulation.n_threads (NEST local_num_threads); "
                        "wire this to Snakemake's `threads` directive")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    main(args.config, n_threads=args.threads)
