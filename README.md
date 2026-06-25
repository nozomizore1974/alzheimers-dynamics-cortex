# column_model

A **self-contained** NEST simulation + analysis pipeline for a single cortical
column (Potjans & Diesmann 2014 microcircuit) with optional amyloid-β (Aβ)
pathology. It runs end-to-end from data files to figures and has **no
dependency on any other project** — copy this folder out and it works on its own
(given `numpy`, `scipy`, `matplotlib`, and NEST 3.x).

## Pipeline

`main.py` runs one flow:

```
read data  →  compute model parameters  →  set up & run simulator
           →  analyse results  →  draw figures
```

## Layout

```
column_model/
├── main.py            # orchestrates the whole flow
├── data_loader.py     # (1) read & process data
├── model_params.py    # (2) compute model parameters (weights, synapses, rates)
├── simulator.py       # (3) NEST set-up & execution
├── analysis.py        # (4a) results analysis (rates, LFP, PSD, wavelet, PLV)
├── plotting.py        # (4b) figures
├── data/              # all input data (edit these, not the code)
│   ├── microcircuit.json    # P&D populations: sizes + connection probabilities (JSON)
│   ├── neuron_params.yaml   # healthy + Aβ neuron params (iaf / aeif), inline units
│   └── config.yaml          # run configuration (model, scale, sim, analysis…), inline units
└── out/               # generated figures + metrics.json
```

## Run

```bash
python main.py                 # uses data/config.yaml
python main.py my_config.yaml  # custom config (YAML or JSON)
```

## Figures produced (in `out/`)

| file | content |
|------|---------|
| `raster_total.png` | total raster of all neurons |
| `raster_subpop.png` | raster coloured by sub-population (layer × healthy/Aβ-hyper/Aβ-supp/Aβ-I) |
| `lfp.png` | kernel LFP proxy: ROI + per-layer |
| `psd.png` | LFP power spectral density (with band shading) |
| `wavelet.png` | Morlet wavelet scalogram (time × frequency) |
| `plv_theta_beta_matrix.png` | layer × layer θ–β n:m PLV |
| `plv_theta_beta_comodulogram.png` | fine-frequency θ–β PLV map of the ROI LFP |

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
  weight×indegree projection → `I_ex + 1.65·I_in` → NN-weighted aggregate),
  z-scored before spectral analysis.
