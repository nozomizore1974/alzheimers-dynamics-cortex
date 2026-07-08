# column_model

A **self-contained** NEST simulation + analysis pipeline for a single cortical
column (Potjans & Diesmann 2014 microcircuit) with optional amyloid-β (Aβ)
pathology. It runs end-to-end from data files to figures and has **no
dependency on any other project** — copy this folder out and it works on its own
(given `numpy`, `scipy`, `matplotlib`, `pyyaml`, `nnmt`, and NEST 3.x; see
`requirements.txt`).

## Pipeline

`main.py` runs one flow:

```
read data  →  compute model parameters  →  set up & run simulator
           →  analyse results  →  draw figures  →  dynamics/stability analysis
```

Heavy stages are **memoised** on disk (`out/<exp>/cache/`, see `cache.py`),
keyed by a hash of the parameters that produced them. Changing a relevant config
value invalidates just the affected entries; delete the cache folder to force a
full recompute.

## Layout

```
column_model/
├── main.py            # orchestrates the whole flow
├── data_loader.py     # (1) read & process data
├── model_params.py    # (2) compute model parameters (weights, synapses, rates)
├── simulator.py       # (3) NEST set-up & execution (also streams raw spikes to .dat)
├── analysis.py        # (4a) results analysis (rates, LFP, PSD, wavelet, PLV)
├── plotting.py        # (4b) figures
├── dynamics.py        # (5) structural + mean-field stability analysis (nnmt)
├── cache.py           # disk memoisation of the sim + heavy analysis products
├── logger.py          # logging helper
├── Snakefile          # batch-run every config under experiments/
├── data/              # all input data (edit these, not the code)
│   ├── microcircuit.json    # P&D populations: sizes + connection probabilities (JSON)
│   ├── neuron_params.yaml   # healthy + Aβ neuron params (iaf / aeif), inline units
│   └── config.yaml          # run configuration (model, scale, sim, analysis…), inline units
├── experiments/       # per-run config YAMLs consumed by the Snakefile
└── out/<exp>/         # generated outputs (see below)
```

## Run

Single run:

```bash
python main.py                 # uses data/config.yaml
python main.py my_config.yaml  # custom config (YAML or JSON)
```

Batch (one output dir per config under `experiments/`, driven by each YAML's
`name:` field):

```bash
snakemake -j4        # run all experiments in parallel
snakemake -n         # dry-run: list what would be built
```

## Outputs (in `out/<exp>/`)

Figures:

| file | content |
|------|---------|
| `raster_total.png` | total raster of all neurons |
| `raster_subpop.png` | raster coloured by sub-population (layer × healthy/Aβ-hyper/Aβ-supp/Aβ-I) |
| `lfp.png` | kernel LFP proxy: raw ROI + z-scored ROI + per-layer |
| `psd.png` | LFP power spectral density (with band shading) |
| `wavelet.png` | Morlet wavelet scalogram (time × frequency) |
| `plv_theta_beta_matrix.png` | layer × layer θ–β n:m PLV |
| `plv_theta_beta_comodulogram.png` | fine-frequency θ–β PLV map of the ROI LFP |
| `dynamics_structural.png` | E/I balance + effective-connectivity spectral radius |
| `dynamics_working_point.png` | per-population mean-field working point |
| `dynamics_stability.png` | async-state stability margin vs frequency |

Data / metadata:

| file | content |
|------|---------|
| `spikes-*.dat` | raw NEST `ascii` spike train (one file per recording thread; columns: sender, time in ms) |
| `metrics.json` | summary metrics (rates, band powers, PLV, dynamics stability) |
| `config_used.json` | the fully resolved config for this run (re-analysis reproducibility) |
| `cache/*.pkl` | memoised simulation + analysis products |

## Key configuration (`data/config.yaml`)

- `model`: `"iaf"` or `"aeif"`; `abeta_ratio`: fraction of Aβ neurons (0 = healthy).
- `N_scale`: population downscaling; `weight_ref_scale`: scale at which weights
  were tuned (recurrent weights ×`weight_ref_scale/N_scale`, the standard linear
  indegree correction that keeps dynamics comparable across scale).
- `synapse.tau_syn_*`, `delay.delay_scale`, `synapse.g_inh`, `synapse.wIE_floor`
  set the network's operating rhythm and the (load-dependent) disinhibition.
- `analysis.plv` sets the θ/β bands, the `nm` harmonic ratio (e.g. `[3,1]` for
  3:1), and the sub-frequency grid for the comodulogram.

## Notes

- The microcircuit (JSON) and neuron parameters (YAML) live in `data/`, so the model can
  be re-parameterised without touching code.
- The LFP proxy is the kernel method (population rate → PSC kernel → signed
  weight×indegree projection → `I_ex + 1.65·I_in` → NN-weighted aggregate). It
  is z-scored (post-warmup statistics) for the ROI output, PSD and wavelet, so
  those all share one convention; PLV is phase-based and uses the raw signal.
- Raw spikes are streamed to `spikes-*.dat` **only when the simulation actually
  runs** — on a cache hit the sim is skipped and the previous `.dat` files are
  reused. Delete `out/<exp>/cache/sim.pkl` to force re-simulation.
- `dynamics.py` predicts, *without running the spiking simulator*, whether the
  asynchronous-irregular state is stable (wiring-only, mean-field working point,
  and linear-stability tiers). It can also be run standalone against a completed
  run to re-analyse it from `config_used.json` + cached rates.
