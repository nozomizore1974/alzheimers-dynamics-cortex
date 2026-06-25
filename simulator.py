"""
simulator.py -- NEST set-up and execution for one cortical column.

Builds the 8 Potjans & Diesmann populations, each split into a healthy subset
and (for E) Abeta "hyperactive" / "suppressed" subsets (and for I a single
Abeta subset), connects them with the precomputed synapse numbers / weights /
delays, adds calibrated Poisson background and a spike recorder, then simulates.

Self-contained: depends only on ``nest`` and numpy.
"""
import numpy as np
import nest


def _mu_sigma_lognorm(mean, rel_sd):
    """(mu, sigma) of a lognormal with given mean and relative SD."""
    return np.log(mean / np.sqrt(rel_sd ** 2 + 1)), np.sqrt(np.log(rel_sd ** 2 + 1))


def _make(model, n, params, extra_I_e=0.0):
    if n < 1:
        return None
    p = dict(params)
    p["I_e"] = p.get("I_e", 0.0) + extra_I_e
    return nest.Create(model, int(n), params=p)


def build_and_simulate(mp, cfg, logger=print):
    """
    Build and run the column described by ``mp`` (ModelParams) and ``cfg``.

    Returns
    -------
    result : dict
        senders, times (np.ndarray spike data),
        pop_gid   : {pop_name: (g0, g1, n)},
        subsets   : list of {pop, kind, g0, g1, n}  (kind in
                    healthy/abeta_hyper/abeta_supp/abeta),
        t_sim, warmup, abeta_ratio.
    """
    sim = cfg["simulation"]
    ab = cfg["abeta"]
    spec = mp.spec
    pop_names, is_exc = mp.pop_names, mp.is_exc

    nest.ResetKernel()
    nest.set_verbosity("M_ERROR")
    nest.SetKernelStatus({"resolution": sim["resolution"], "local_num_threads": sim["n_threads"],
                          "rng_seed": sim["master_seed"], "overwrite_files": True})

    # ---------- populations (healthy / Abeta subsets) ----------
    pops, pop_gid, subsets = {}, {}, []
    load = mp.abeta_ratio
    for k, name in enumerate(pop_names):
        E = is_exc[k]
        model = spec["model_E"] if E else spec["model_I"]
        p_h = spec["E_healthy"] if E else spec["I_healthy"]
        p_a = spec["E_abeta"] if E else spec["I_abeta"]
        n_tot = int(mp.N[k])
        n_ab = int(round(n_tot * load))
        n_h = n_tot - n_ab
        parts = []
        nc = _make(model, n_h, p_h)
        if nc is not None:
            parts.append(("healthy", nc))
        if E:
            n_hi = int(round(n_ab * ab["hyper_frac"]))
            n_su = n_ab - n_hi
            for kind, nn, ie in (("abeta_hyper", n_hi, ab["I_e_hyper"]),
                                 ("abeta_supp", n_su, ab["I_e_supp"])):
                c = _make(model, nn, p_a, extra_I_e=ie)
                if c is not None:
                    parts.append((kind, c))
        else:
            c = _make(model, n_ab, p_a)
            if c is not None:
                parts.append(("abeta", c))

        merged = None
        for kind, nc in parts:
            merged = nc if merged is None else merged + nc
            subsets.append(dict(pop=name, kind=kind, g0=nc[0].global_id,
                                g1=nc[-1].global_id, n=len(nc)))
        merged.V_m = nest.random.normal(mean=sim["V0_mean"], std=sim["V0_sd"])
        pops[name] = merged
        pop_gid[name] = (merged[0].global_id, merged[-1].global_id, len(merged))

    # ---------- recurrent connections (P&D) ----------
    res = sim["resolution"]
    for i, tgt in enumerate(pop_names):
        for j, src in enumerate(pop_names):
            n_syn = int(mp.n_syn[i, j])
            w = mp.W[i, j]
            if n_syn < 1 or w == 0.0:
                continue
            d_mean = mp.delay_e if is_exc[j] else mp.delay_i
            mu_d, sig_d = _mu_sigma_lognorm(d_mean, mp.delay_rel)
            w_sd = abs(w) * mp.PSP_rel
            syn_spec = {"synapse_model": "static_synapse",
                        "delay": nest.math.max(nest.random.lognormal(mean=mu_d, std=sig_d), res)}
            if w >= 0:
                syn_spec["weight"] = nest.math.max(nest.random.normal(mean=w, std=w_sd), 0.0)
            else:
                syn_spec["weight"] = nest.math.min(nest.random.normal(mean=w, std=w_sd), 0.0)
            nest.Connect(pops[src], pops[tgt],
                         {"rule": "fixed_total_number", "N": n_syn,
                          "allow_autapses": False, "allow_multapses": True}, syn_spec)

    # ---------- external Poisson drive ----------
    for j, name in enumerate(pop_names):
        pg = nest.Create("poisson_generator", params={"rate": float(mp.rate_ext[j])})
        nest.Connect(pg, pops[name], "all_to_all",
                     syn_spec={"synapse_model": "static_synapse",
                               "weight": float(mp.w_ext[j]), "delay": res})

    # ---------- recorder + simulate ----------
    sr = nest.Create("spike_recorder")
    for nc in pops.values():
        nest.Connect(nc, sr)
    if logger:
        logger(f"[simulate] {sum(v[2] for v in pop_gid.values())} neurons, "
               f"abeta_ratio={load}, t_sim={sim['t_sim']} ms ...")
    nest.Simulate(sim["t_sim"])
    ev = sr.get("events")
    return dict(senders=np.asarray(ev["senders"]), times=np.asarray(ev["times"]),
                pop_gid=pop_gid, subsets=subsets,
                t_sim=sim["t_sim"], warmup=sim["warmup"], abeta_ratio=load)
