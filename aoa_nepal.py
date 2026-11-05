"""
Area of Applicability (AOA) for a RF model.

Generates a grid that specifies, for each Nepal pixel, whether a prediction
would be an interpolation (within the training feature space) or an extrapolation.

Method: Meyer & Pebesma (2021), Methods in Ecology and Evolution,
"Predicting into unknown space? Estimating the area of applicability of
spatial prediction models".

"""

import numpy as np
import pandas as pd
import rasterio
from scipy.spatial.distance import pdist
from scipy.stats import chi2
from sklearn.neighbors import NearestNeighbors

CHUNK = 200_000                 
NODATA = -1


# ------------------------------------------------------------------ Laden
def load_nepal(nepal_paths, nodata=NODATA):
    """Load Nepal predictor rasters.

    nepal_paths : A list of paths, in EXACTLY the same order as
                  predictor_names. Do not derive them from filenames -- the
                  order of the columns must match the model.

    Check raster alignment.
    """
    stack, profile, shape, transform = [], None, None, None
    for path in nepal_paths:
        with rasterio.open(path) as src:
            if profile is None:
                profile, shape, transform = src.profile.copy(), src.shape, src.transform
            else:
                assert src.shape == shape, f"Shape mismatch: {path}"
                assert np.allclose(
                    [src.transform.a, src.transform.b, src.transform.c,
                     src.transform.d, src.transform.e, src.transform.f],
                    [transform.a, transform.b, transform.c,
                     transform.d, transform.e, transform.f], atol=1e-6), \
                    f"Transform mismatch: {path}"
            stack.append(src.read(1).astype(np.float32))

    arr = np.stack(stack, axis=-1).reshape(-1, len(nepal_paths))
    valid = (arr != nodata).all(axis=1) & np.isfinite(arr).all(axis=1)
    print(f"Nepal: {shape[0]}x{shape[1]} = {shape[0]*shape[1]:,} Zellen, "
          f"davon {valid.sum():,} gueltig ({100*valid.mean():.1f}%)")
    return arr, valid, profile, shape


# -------------------------------------------------------------------- AOA
def compute_aoa(train_X, new_X, weights=None, sample_for_dbar=2000, seed=42):
    """Dissimilarity Index and AOA-Threshold by Meyer & Pebesma (2021).

    1. Predictors with Training Means/SDs scale
    2. optional with Variable Importance weights
    3. DI = Distance to nearest Training Point / average Training Distance
    4. Threshold from the distribution of Training-DI (Leave-one-out):
       Q75 + 1.5 * IQR
    """
    rng = np.random.default_rng(seed)

    mu = train_X.mean(axis=0)
    sd = train_X.std(axis=0)
    sd[sd < 1e-9] = 1.0

    w = np.ones(train_X.shape[1]) if weights is None else np.asarray(weights, float)
    w = w / w.sum() * len(w)

    tr = (train_X - mu) / sd * np.sqrt(w)

    # Average pairwise distance in the training set (normalization)
    idx = rng.choice(len(tr), min(sample_for_dbar, len(tr)), replace=False)
    d_bar = pdist(tr[idx]).mean()

    nn = NearestNeighbors(n_neighbors=2, n_jobs=-1).fit(tr)

    # Training-DI: Distance to the nearest OTHER training point
    d_tr, _ = nn.kneighbors(tr)
    di_train = d_tr[:, 1] / d_bar
    q25, q75 = np.percentile(di_train, [25, 75])
    threshold = q75 + 1.5 * (q75 - q25)

    # DI of the new points, blockwise due to memory constraints
    di_new = np.empty(len(new_X), dtype=np.float32)
    for s in range(0, len(new_X), CHUNK):
        block = (new_X[s:s + CHUNK] - mu) / sd * np.sqrt(w)
        d, _ = nn.kneighbors(block, n_neighbors=1)
        di_new[s:s + CHUNK] = d[:, 0] / d_bar

    return di_new, threshold, di_train


# ------------------------------------------------- univariate Abdeckung
def univariate_coverage(train_X, new_X, predictor_names, q=(1, 99)):
    """For each predictor: Proportion of Nepal pixels outside the training range.
    Simplest diagnosis, ignores the multivariate structure."""
    rows = []
    for k, name in enumerate(predictor_names):
        lo_a, hi_a = train_X[:, k].min(), train_X[:, k].max()
        lo_q, hi_q = np.percentile(train_X[:, k], q)
        rows.append({
            "predictor": name,
            "train_min": lo_a, "train_max": hi_a,
            "outside_range_pct": 100 * np.mean(
                (new_X[:, k] < lo_a) | (new_X[:, k] > hi_a)),
            "outside_p1_p99_pct": 100 * np.mean(
                (new_X[:, k] < lo_q) | (new_X[:, k] > hi_q)),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------- schreiben
def write_mask(values, valid, shape, profile, out_path, dtype="float32"):
    """Write values back to raster layout, invalid pixels to NODATA."""
    full = np.full(shape[0] * shape[1], NODATA, dtype=np.float32)
    full[valid] = values
    profile.update(dtype=dtype, count=1, nodata=NODATA, compress="lzw")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(full.reshape(shape).astype(dtype), 1)
    print(f"written: {out_path}")
