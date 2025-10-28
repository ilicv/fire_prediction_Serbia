# -*- coding: utf-8 -*-
# Balanced sampler with two passes
import os, csv, glob, re, random
from osgeo import gdal
import numpy as np

LABEL_BASE = "Srbija pozari 400m grid"
OUTPUT_CSV = "final_dataset_from_resampled_sampled.csv"

CATEGORICAL_BASES = {"Aspect", "Land use"}
LAND_USE_CLOUDS_CLASS = 10

# Sampling params
GRID_STEP = 3                 # sample every 3rd pixel in both axes
TARGET_NEG_PER_POS = 1.5      # desired negatives per positive (1.0 => roughly 50/50)
POSITIVE_SAMPLE_FRACTION = 1.0  # keep all positives by default; set <1.0 to downsample positives too
RANDOM_SEED = 1337

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def open_band(path):
    ds = gdal.Open(path)
    if ds is None:
        raise FileNotFoundError(path)
    b = ds.GetRasterBand(1)
    return ds, b.ReadAsArray(), b.GetNoDataValue()

def find_resampled():
    rasters = {}
    for p in glob.glob("*_resampled.tif"):
        base = re.sub(r"_resampled$", "", os.path.splitext(os.path.basename(p))[0], flags=re.IGNORECASE)
        rasters[base] = p
    return rasters

def is_nan_or_nodata(val, nd):
    if nd is not None and val == nd:
        return True
    try:
        return np.isnan(val)
    except Exception:
        return False

def _norm(s):
    s = s.lower().replace("_"," ")
    return re.sub(r"\s+"," ",s).strip()

def resolve_label_path(rasters, preferred_base):
    if preferred_base in rasters:
        return preferred_base, rasters[preferred_base]
    want = _norm(preferred_base)
    for k,p in rasters.items():
        if _norm(k) == want:
            return k,p
    cands = [(k,p) for k,p in rasters.items() if "srbija" in _norm(k) and "grid" in _norm(k)]
    if len(cands)==1:
        return cands[0]
    if len(cands)>1:
        for k,p in cands:
            if "pozari" in _norm(k):
                return k,p
        return cands[0]
    return None,None

def main():
    print("Scanning for *_resampled.tif ...")
    rasters = find_resampled()
    if not rasters:
        raise RuntimeError("No *_resampled.tif here.")

    label_base, label_path = resolve_label_path(rasters, LABEL_BASE)
    if not label_path or not os.path.exists(label_path):
        print("Available rasters:", ", ".join(sorted(rasters.keys())))
        raise FileNotFoundError("Label raster not found (expected '{}_resampled.tif')".format(LABEL_BASE))

    print("Using label:", label_path, "(base='{}')".format(label_base))
    ref_ds, ref_arr, ref_nd = open_band(label_path)
    rows, cols = ref_ds.RasterYSize, ref_ds.RasterXSize
    gt = ref_ds.GetGeoTransform()

    bases = sorted([b for b in rasters.keys() if os.path.abspath(rasters[b]) != os.path.abspath(label_path)])
    categorical_bases = [b for b in bases if b in CATEGORICAL_BASES]
    numeric_bases     = [b for b in bases if b not in CATEGORICAL_BASES]
    print("Categorical:", categorical_bases)
    print("Numeric    :", numeric_bases)

    # Load features
    feature_arrays, feature_nd = {}, {}
    for b in bases:
        ds, arr, nd = open_band(rasters[b])
        if arr.shape != ref_arr.shape:
            raise RuntimeError("Shape mismatch: {} vs label grid".format(rasters[b]))
        feature_arrays[b] = arr
        feature_nd[b]     = nd

    # Load label
    _, y_arr, y_nd = open_band(label_path)

    # Prepare OHE schema (exclude clouds from emitted columns)
    ohe_values = {}
    for cb in categorical_bases:
        arr = feature_arrays[cb]
        nd  = feature_nd[cb]
        mask = np.ones(arr.shape, dtype=bool)
        if nd is not None: mask &= (arr != nd)
        vals = arr[mask].astype(np.int64)
        uniq = np.unique(vals)
        if cb == "Land use":
            uniq = uniq[uniq != LAND_USE_CLOUDS_CLASS]
        ohe_values[cb] = sorted(uniq.tolist())

    # ---------- PASS 1: count valid positives / negatives ----------
    print(f"Pass 1 (count with GRID_STEP={GRID_STEP}) ...")
    pos_valid = 0
    neg_valid = 0
    for r in range(0, rows, GRID_STEP):
        if r % 200 == 0:
            print("  Count row {}/{}".format(r+1, rows))
        for c in range(0, cols, GRID_STEP):
            # label
            yv = y_arr[r, c]
            if is_nan_or_nodata(yv, y_nd):
                continue
            try:
                yv = 1 if float(yv) > 0 else 0
            except Exception:
                continue

            # skip cloud pixels early (needs Land use)
            if "Land use" in feature_arrays:
                lu = feature_arrays["Land use"][r, c]
                if not is_nan_or_nodata(lu, feature_nd["Land use"]):
                    if int(round(float(lu))) == LAND_USE_CLOUDS_CLASS:
                        continue

            # check numeric features NoData
            bad = False
            for nb in numeric_bases:
                v = feature_arrays[nb][r, c]
                if is_nan_or_nodata(v, feature_nd[nb]):
                    bad = True
                    break
            if bad: continue

            # check categorical NoData (but not OHE emission)
            for cb in categorical_bases:
                v = feature_arrays[cb][r, c]
                if is_nan_or_nodata(v, feature_nd[cb]):
                    bad = True
                    break
            if bad: continue

            if yv == 1: pos_valid += 1
            else:       neg_valid += 1

    print(f"Counts after filters → positives: {pos_valid:,}, negatives: {neg_valid:,}")

    # decide negative keep fraction to hit TARGET_NEG_PER_POS
    if pos_valid == 0:
        neg_keep_frac = 0.0
    elif neg_valid == 0:
        neg_keep_frac = 0.0
    else:
        desired_negs = TARGET_NEG_PER_POS * pos_valid
        neg_keep_frac = min(1.0, desired_negs / neg_valid)

    print(f"Computed negative keep fraction: {neg_keep_frac:.4f}  (target neg/pos = {TARGET_NEG_PER_POS})")
    print(f"Positive keep fraction         : {POSITIVE_SAMPLE_FRACTION:.4f}")

    # ---------- PASS 2: write sampled CSV ----------
    header = ["X", "Y"] + numeric_bases
    for cb in categorical_bases:
        header += [f"{cb}=={v}" for v in ohe_values[cb]]
    header += ["label"]

    print("Writing ->", OUTPUT_CSV)
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

        pos_keep = neg_keep = 0
        for r in range(0, rows, GRID_STEP):
            if r % 200 == 0:
                print("  Write row {}/{}".format(r+1, rows))
            for c in range(0, cols, GRID_STEP):
                # label
                yv = y_arr[r, c]
                if is_nan_or_nodata(yv, y_nd):
                    continue
                try:
                    yv = 1 if float(yv) > 0 else 0
                except Exception:
                    continue

                # sample by class
                if yv == 1:
                    if random.random() > POSITIVE_SAMPLE_FRACTION:
                        continue
                else:
                    if random.random() > neg_keep_frac:
                        continue

                # numeric features
                row_vals = []
                bad = False
                for nb in numeric_bases:
                    v = feature_arrays[nb][r, c]
                    if is_nan_or_nodata(v, feature_nd[nb]):
                        bad = True
                        break
                    try:
                        iv = int(v)
                        if abs(v - iv) < 1e-6:
                            row_vals.append(iv)
                        else:
                            row_vals.append(round(float(v), 4))
                    except Exception:
                        row_vals.append(round(float(v), 4))
                if bad:
                    continue

                # categorical OHE (skip clouds rows)
                ohe_bits = []
                for cb in categorical_bases:
                    v = feature_arrays[cb][r, c]
                    if is_nan_or_nodata(v, feature_nd[cb]):
                        bad = True
                        break
                    v = int(round(float(v)))
                    if cb == "Land use" and v == LAND_USE_CLOUDS_CLASS:
                        bad = True
                        break
                    ohe_bits.extend([1 if v == k else 0 for k in ohe_values[cb]])
                if bad:
                    continue

                # coords
                x = gt[0] + c * gt[1] + r * gt[2]
                y = gt[3] + c * gt[4] + r * gt[5]

                w.writerow([x, y] + row_vals + ohe_bits + [yv])
                if yv == 1: pos_keep += 1
                else:       neg_keep += 1

    total = pos_keep + neg_keep
    ratio = (neg_keep / pos_keep) if pos_keep else float('inf')
    print(f"✅ Wrote: {OUTPUT_CSV}")
    print(f"   kept positives: {pos_keep:,}")
    print(f"   kept negatives: {neg_keep:,}")
    print(f"   total rows    : {total:,}")
    print(f"   achieved neg/pos ratio: {ratio:.3f} (target {TARGET_NEG_PER_POS})")

if __name__ == "__main__":
    main()
