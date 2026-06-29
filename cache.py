"""
cache.py -- tiny disk cache so expensive stages are computed once and reused.

The NEST simulation and the heavy analysis transforms (wavelet scalogram,
theta-beta PLV matrix / comodulogram) are the slow parts of the pipeline.
Each is wrapped with :func:`cached`, which stores its result on disk keyed by a
stable hash of the parameters that produced it.  On a re-run the cache is read
back instead of recomputing; change a relevant config value and the hash no
longer matches, so the stale entry is bypassed and recomputed automatically --
no manual cleanup needed.

This mirrors ../Adnest's reuse design (``dicthash`` parameter hashing +
``try: read_pickle / except: compute; to_pickle``), implemented with the
standard library only so the project stays self-contained.
"""
import os
import json
import pickle
import hashlib

import numpy as np


def _jsonify(o):
    """Make numpy scalars/arrays JSON-serialisable for hashing."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(f"param_hash: unhashable type {type(o)!r}")


def param_hash(*objs):
    """Stable 16-char hash of JSON-able objects (dict key order ignored)."""
    blob = json.dumps(objs, sort_keys=True, default=_jsonify, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def cached(path, key, compute, logger=print):
    """Return ``compute()``, memoised at ``path`` under ``key``.

    The result is pickled as ``{"key": key, "value": ...}``.  It is recomputed
    when the file is missing, unreadable, or its stored key differs from
    ``key`` -- so a parameter change invalidates the entry on its own.

    Parameters
    ----------
    path : str
        Destination ``.pkl`` file (its directory is created if needed).
    key : str
        Parameter hash identifying this result (see :func:`param_hash`).
    compute : callable
        Zero-argument function producing the (picklable) value on a miss.
    """
    name = os.path.basename(path)
    if os.path.exists(path):
        try:
            with open(path, "rb") as fh:
                blob = pickle.load(fh)
            if blob.get("key") == key:
                if logger:
                    logger(f"[cache] hit  {name}")
                return blob["value"]
            if logger:
                logger(f"[cache] stale {name} -> recomputing")
        except Exception:
            if logger:
                logger(f"[cache] bad  {name} -> recomputing")

    elif logger:
        logger(f"[cache] miss {name} -> computing")

    value = compute()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump({"key": key, "value": value}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)  # atomic publish; never leave a half-written cache
    return value
