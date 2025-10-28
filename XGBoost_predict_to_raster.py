# -*- coding: utf-8 -*-
# XGBoost_predict_to_raster.py — use trained XGBoost model to produce prob GeoTIFF
import os, re, glob
import numpy as np
import pandas as pd
from osgeo import gdal
import xgboost as xgb

# ====== INPUTS ======
MODEL_JSON       = "XGBoost_fire_model.json"   # saved by XGBoost_Train.py
FEATURE_ORDER_CSV= "feature_columns.csv"       # from prepare_ddn_dataset.py
MINMAX_CSV       = "min_max_values.csv"        # numeric scaling (same prep)
LABEL_BASE       = "Srbija pozari 400m grid"   # for georeference
OUT_TIF          = "XGBoost_pred_fire_prob.tif"
OUT_RGB          = "XGBoost_pred_fire_prob_RGB.tif"
OUT_NODATA       = -9999.0
TILE             = 1024                        # tile size

# Categorical one-hot bases used in CSV
CATEGORICAL_BASES = {"Aspect", "Land use"}
LAND_USE_CLOUDS_CLASS = 10

# ---------- helpers ----------
def find_resampled():
    rasters = {}
    for p in glob.glob("*_resampled.tif"):
        base = re.sub(r"_resampled$", "", os.path.splitext(os.path.basename(p))[0], flags=re.IGNORECASE)
        rasters[base] = p
    return rasters

def open_band(path):
    ds = gdal.Open(path)
    if ds is None:
        raise FileNotFoundError(path)
    band = ds.GetRasterBand(1)
    return ds, band.ReadAsArray(), band.GetNoDataValue()

def _norm(s):
    s = s.lower().replace("_", " ")
    return re.sub(r"\s+", " ", s).strip()

def resolve_label_path(rasters, preferred_base):
    if preferred_base in rasters:
        return preferred_base, rasters[preferred_base]
    want = _norm(preferred_base)
    for k, p in rasters.items():
        if _norm(k) == want:
            return k, p
    cands = [(k,p) for k,p in rasters.items() if "srbija" in _norm(k) and "grid" in _norm(k)]
    if len(cands) == 1: return cands[0]
    if len(cands) > 1:
        for k,p in cands:
            if "pozari" in _norm(k): return k,p
        return cands[0]
    return None, None

def parse_ohe_schema(ohe_cols):
    schema = {}
    for c in ohe_cols:
        if "==" not in c: continue
        base, val = c.split("==", 1)
        base = base.strip()
        v = int(val.strip())
        schema.setdefault(base, set()).add(v)
    for k in schema:
        schema[k] = sorted(schema[k])
    return schema

def minmax_scale_inline(name, a, minmax):
    if name in minmax:
        mn, mx = minmax[name]
        if mx is None or mn is None or mx == mn:
            return np.zeros_like(a, dtype=np.float32)
        return ((a - mn) / (mx - mn)).astype(np.float32)
    return a.astype(np.float32)


def probs_to_rgb(p, valid_mask):
    """
    Map probabilities [0,1] to RGB:
      [0, 0.5]: green(0,255,0) -> yellow(255,255,0)
      [0.5, 1]: yellow(255,255,0) -> red(255,0,0)
    invalid -> 0 (black)
    """
    import numpy as np
    q = np.clip(p, 0.0, 1.0).astype(np.float32)
    r = np.zeros_like(q, dtype=np.float32)
    g = np.zeros_like(q, dtype=np.float32)
    b = np.zeros_like(q, dtype=np.float32)
    mid = 0.5
    low = (q <= mid)
    high = ~low
    if low.any():
        t = (q[low] / mid)
        r[low] = 255.0 * t
        g[low] = 100.0 + (155.0 * t)
        b[low] = 0.0
    if high.any():
        t = (q[high] - mid) / (1.0 - mid)
        r[high] = 255.0
        g[high] = 255.0 * (1.0 - t)
        b[high] = 0.0
    r[~valid_mask] = 0.0
    g[~valid_mask] = 0.0
    b[~valid_mask] = 0.0
    return r.astype(np.uint8), g.astype(np.uint8), b.astype(np.uint8)

def main():
    # --- load feature order & minmax ---
    if not os.path.exists(FEATURE_ORDER_CSV):
        raise FileNotFoundError(FEATURE_ORDER_CSV)
    feat_order = pd.read_csv(FEATURE_ORDER_CSV)["feature"].tolist()

    minmax_df = pd.read_csv(MINMAX_CSV) if os.path.exists(MINMAX_CSV) else pd.DataFrame(columns=["feature","min","max"])
    minmax = {row["feature"]: (row["min"], row["max"]) for _, row in minmax_df.iterrows()}

    ohe_cols = [c for c in feat_order if "==" in c]
    num_cols = [c for c in feat_order if "==" not in c]
    ohe_schema = parse_ohe_schema(ohe_cols)

    # --- rasters & reference geotransform ---
    rasters = find_resampled()
    if not rasters:
        raise RuntimeError("No *_resampled.tif in this folder.")

    label_base, label_path = resolve_label_path(rasters, LABEL_BASE)
    if not label_path:
        raise FileNotFoundError("Label/reference raster (Srbija pozari 400m grid_resampled.tif) not found.")

    ref_ds = gdal.Open(label_path)
    gt = ref_ds.GetGeoTransform()
    proj = ref_ds.GetProjection()
    rows, cols = ref_ds.RasterYSize, ref_ds.RasterXSize

    # load numeric sources
    num_arrays, num_nodata = {}, {}
    for name in num_cols:
        if name not in rasters:
            raise FileNotFoundError(f"Missing numeric raster for feature '{name}'")
        _, arr, nd = open_band(rasters[name])
        if arr.shape != (rows, cols):
            raise RuntimeError(f"Shape mismatch: {rasters[name]}")
        num_arrays[name] = arr.astype(np.float32, copy=False)
        num_nodata[name] = nd

    # load categorical sources used by schema
    cat_arrays, cat_nodata = {}, {}
    for base in ohe_schema.keys():
        if base not in rasters:
            raise FileNotFoundError(f"Missing categorical raster for '{base}'")
        _, arr, nd = open_band(rasters[base])
        if arr.shape != (rows, cols):
            raise RuntimeError(f"Shape mismatch: {rasters[base]}")
        cat_arrays[base] = arr
        cat_nodata[base] = nd

    # --- load XGBoost model ---
    if not os.path.exists(MODEL_JSON):
        raise FileNotFoundError(MODEL_JSON)
    model = xgb.Booster()
    model.load_model(MODEL_JSON)
    # (predictor/tree_method are embedded; if not, XGB will auto choose)

    # --- prepare output raster ---
    drv = gdal.GetDriverByName("GTiff")
    if os.path.exists(OUT_TIF):
        drv.Delete(OUT_TIF)
    out_ds = drv.Create(OUT_TIF, cols, rows, 1, gdal.GDT_Float32,
                        options=["COMPRESS=LZW", "TILED=YES"])
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)
    out_band = out_ds.GetRasterBand(1)


    out_band.SetNoDataValue(float(OUT_NODATA))

    # NEW: RGB visualization GeoTIFF (uint8, 3 bands)
    if os.path.exists(OUT_RGB):
        drv.Delete(OUT_RGB)
    rgb_ds = drv.Create(OUT_RGB, cols, rows, 3, gdal.GDT_Byte,
                        options=["COMPRESS=LZW", "TILED=YES", "INTERLEAVE=PIXEL"])
    rgb_ds.SetGeoTransform(gt)
    rgb_ds.SetProjection(proj)
    rgb_r = rgb_ds.GetRasterBand(1); rgb_g = rgb_ds.GetRasterBand(2); rgb_b = rgb_ds.GetRasterBand(3)
    for b in (rgb_r, rgb_g, rgb_b):
        b.SetNoDataValue(0)

    # --- tilewise inference ---
    for r0 in range(0, rows, TILE):
        r1 = min(rows, r0 + TILE); h = r1 - r0
        for c0 in range(0, cols, TILE):
            c1 = min(cols, c0 + TILE); w = c1 - c0

            valid = np.ones((h, w), dtype=bool)

            # numeric stack (scaled)
            num_stack = []
            for name in num_cols:
                tile = num_arrays[name][r0:r1, c0:c1]
                nd = num_nodata[name]
                if nd is not None: valid &= (tile != nd)
                tile = minmax_scale_inline(name, tile, minmax)
                num_stack.append(tile)

            # categorical → OHE per schema (skip clouds)
            ohe_map = {}
            for base, cats in ohe_schema.items():
                tile = cat_arrays[base][r0:r1, c0:c1]
                nd = cat_nodata[base]
                if nd is not None: valid &= (tile != nd)
                if base == "Land use":
                    valid &= (tile.astype(np.int32) != LAND_USE_CLOUDS_CLASS)
                as_int = tile.astype(np.int32)
                for k in cats:
                    ohe_map[(base, k)] = (as_int == k).astype(np.float32)

            if (len(num_stack) == 0) and (len(ohe_map) == 0):
                out_band.WriteArray(np.full((h, w), np.float32(OUT_NODATA)), xoff=c0, yoff=r0)
                rgb_r.WriteArray(np.zeros((h, w), dtype=np.uint8), xoff=c0, yoff=r0)
                rgb_g.WriteArray(np.zeros((h, w), dtype=np.uint8), xoff=c0, yoff=r0)
                rgb_b.WriteArray(np.zeros((h, w), dtype=np.uint8), xoff=c0, yoff=r0)
                continue

            # build feature block EXACTLY as in feature_columns.csv: [num_cols..., ohe_cols...]
            X_tiles = []
            for name in num_cols:
                X_tiles.append(num_stack[num_cols.index(name)])
            for col in ohe_cols:
                base, val = col.split("==", 1)
                k = int(val)
                X_tiles.append(ohe_map[(base, k)])

            X_block = np.stack(X_tiles, axis=-1)     # (h, w, F)
            X_flat = X_block.reshape(-1, X_block.shape[-1])

            vmask = valid.reshape(-1)
            Xv = X_flat[vmask]
            if Xv.size == 0:
                out_band.WriteArray(np.full((h, w), np.float32(OUT_NODATA)), xoff=c0, yoff=r0)
                rgb_r.WriteArray(np.zeros((h, w), dtype=np.uint8), xoff=c0, yoff=r0)
                rgb_g.WriteArray(np.zeros((h, w), dtype=np.uint8), xoff=c0, yoff=r0)
                rgb_b.WriteArray(np.zeros((h, w), dtype=np.uint8), xoff=c0, yoff=r0)
                continue

            # XGBoost expects DMatrix; set feature_names=feat_order to align
            dmat = xgb.DMatrix(Xv, feature_names=feat_order)
            probs = model.predict(dmat)  # 'binary:logistic' → probabilities

            # write back
            out_tile = np.full((h*w,), np.float32(OUT_NODATA), dtype=np.float32)
            out_tile[vmask] = probs.astype(np.float32)
            out_tile = out_tile.reshape(h, w)
            out_band.WriteArray(out_tile, xoff=c0, yoff=r0)

            # Write RGB visualization for this tile
            r8, g8, b8 = probs_to_rgb(out_tile, valid)
            rgb_r.WriteArray(r8, xoff=c0, yoff=r0)
            rgb_g.WriteArray(g8, xoff=c0, yoff=r0)
            rgb_b.WriteArray(b8, xoff=c0, yoff=r0)

    out_band.FlushCache()
    out_ds = None
    rgb_r.FlushCache(); rgb_g.FlushCache(); rgb_b.FlushCache(); rgb_ds = None
    print("✅ Wrote raster probabilities →", os.path.abspath(OUT_TIF))
    print("🎨 Wrote RGB visualization  →", os.path.abspath(OUT_RGB))

if __name__ == "__main__":
    main()