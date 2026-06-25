"""
data_loader.py -- read & process input data for the column model.

Self-contained: depends only on the data files in ``data/``:
  * microcircuit.json     -- microcircuit definition (kept as JSON)
  * neuron_params.yaml    -- neuron parameter sets (YAML, inline-commented)
  * config.yaml           -- run configuration (YAML, inline-commented)
No dependency on any external project. The microcircuit / neuron-parameter /
config values are *data*, not hard-coded, so they can be swapped out without
touching the code.
"""
import os
import json
import numpy as np
import yaml

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _load_json(name, data_dir=DATA_DIR):
    with open(os.path.join(data_dir, name)) as f:
        return json.load(f)


def load_config(path=None, data_dir=DATA_DIR):
    """Load the run configuration (YAML). ``path`` overrides the default file."""
    if path is None:
        path = os.path.join(data_dir, "config.yaml")
    return _load_yaml(path)


def load_microcircuit(data_dir=DATA_DIR):
    """
    Load the Potjans & Diesmann microcircuit definition (JSON).

    Returns
    -------
    dict with keys: pop_names (list[str]), layers (list[str]),
        N_full (np.ndarray[int]), conn_probs (np.ndarray[8,8], target x source),
        is_exc (np.ndarray[bool]), pop_layer (list[str]).
    """
    mc = _load_json("microcircuit.json", data_dir)
    pop_names = list(mc["pop_names"])
    return {
        "pop_names": pop_names,
        "layers": list(mc["layers"]),
        "N_full": np.asarray(mc["N_full"], dtype=int),
        "conn_probs": np.asarray(mc["conn_probs"], dtype=float),
        "is_exc": np.array([p.endswith("E") for p in pop_names]),
        "pop_layer": [p[:-1] for p in pop_names],   # 'L23E' -> 'L23'
    }


def load_neuron_params(data_dir=DATA_DIR):
    """Load the healthy + Abeta neuron parameter sets for all models (YAML)."""
    return _load_yaml(os.path.join(data_dir, "neuron_params.yaml"))


def scaled_neuron_numbers(N_full, N_scale):
    """Down/upscale population sizes; at least 1 neuron per population."""
    return np.maximum(1, np.round(np.asarray(N_full) * N_scale).astype(int))


def build_spec(neuron_params, model, tau_syn_ex=None, tau_syn_in=None):
    """
    Assemble the neuron specification for one model family, applying optional
    global synaptic-time-constant overrides (used to place the network rhythm
    in a chosen frequency band).

    Returns
    -------
    dict with model_E, model_I and the four parameter dicts E_healthy,
    E_abeta, I_healthy, I_abeta (deep-copied, with overrides applied).
    """
    if model not in neuron_params:
        raise KeyError(f"model '{model}' not in neuron_params (have {list(neuron_params)})")
    src = neuron_params[model]

    def _ovr(p):
        p = dict(p)
        if tau_syn_ex is not None:
            p["tau_syn_ex"] = tau_syn_ex
        if tau_syn_in is not None:
            p["tau_syn_in"] = tau_syn_in
        return p

    return {
        "name": model,
        "model_E": src["model_E"],
        "model_I": src["model_I"],
        "E_healthy": _ovr(src["E_healthy"]),
        "E_abeta":   _ovr(src["E_abeta"]),
        "I_healthy": _ovr(src["I_healthy"]),
        "I_abeta":   _ovr(src["I_abeta"]),
    }
