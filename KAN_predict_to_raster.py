# -*- coding: utf-8 -*-
# KAN__predict_to_raster.py  use trained KAN model to produce probability GeoTIFF
import os, re, glob
import numpy as np
import pandas as pd
from osgeo import gdal
import torch
import torch.nn as nn

# ====== CONFIG ======
MODEL_WEIGHTS      = "KAN_best_model.pth"          # saved by KAN_Train.py
FEATURE_ORDER_CSV  = "feature_columns.csv"         # from prepare_ddn_dataset.py
MINMAX_CSV         = "min_max_values.csv"          # numeric scaling (same prep)
LABEL_BASE         = "Srbija pozari 400m grid"     # for georeference
OUT_TIF            = "KAN_pred_fire_prob.tif"
OUT_RGB            = "KAN_pred_fire_prob_RGB.tif"
OUT_NODATA         = -9999.0
TILE               = 1024                           # tile size (adjust for RAM/VRAM)

# Categorical one-hot bases used in CSV
CATEGORICAL_BASES = {"Aspect", "Land use"}
LAND_USE_CLOUDS_CLASS = 10

# ====== MODEL (must match KAN_Train.py) ======

# ====== Model (Residual MLP with BN + SiLU) ======
class _Block(nn.Module):
    def __init__(self, in_dim, out_dim, p_drop=0.15):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.bn1  = nn.BatchNorm1d(out_dim)
        self.act  = nn.SiLU()
        self.drop = nn.Dropout(p_drop)
        self.lin2 = nn.Linear(out_dim, out_dim)
        self.bn2  = nn.BatchNorm1d(out_dim)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    def forward(self, x):
        y = self.lin1(x); y = self.bn1(y); y = self.act(y); y = self.drop(y)
        y = self.lin2(y); y = self.bn2(y)
        return self.act(y + self.proj(x))

class KANModel(nn.Module):
    """Residual MLP backbone (no final sigmoid; BCEWithLogits at train)."""
    def __init__(self, input_size):
        super().__init__()
        widths = [256, 128, 64]
        d = input_size
        layers = []
        for w in widths:
            layers.append(_Block(d, w, p_drop=0.15))
            d = w
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)  # logits
    def forward(self, x):
        return self.head(self.backbone(x))

# ====== HELPERS ======
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

def parse_ohe_schema(ohe_cols):
    schema = {}
    for c in ohe_cols:
        if "==" not in c: 
            continue
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
    # ---- load feature order & scaling ----
    if not os.path.exists(FEATURE_ORDER_CSV):
        raise FileNotFoundError(FEATURE_ORDER_CSV)
    feat_order = pd.read_csv(FEATURE_ORDER_CSV)["feature"].tolist()

    minmax_df = pd.read_csv(MINMAX_CSV) if os.path.exists(MINMAX_CSV) else pd.DataFrame(columns=["feature","min","max"])
    minmax = {row["feature"]: (row["min"], row["max"]) for _, row in minmax_df.iterrows()}

    ohe_cols = [c for c in feat_order if "==" in c]
    num_cols = [c for c in feat_order if "==" not in c]
    ohe_schema = parse_ohe_schema(ohe_cols)  # e.g., {"Aspect":[1..9], "Land use":[1,2,4,5,7,8,9,11]}

    # ---- rasters & reference ----
    rasters = find_resampled()
    if not rasters:
        raise RuntimeError("No *_resampled.tif in this folder.")

    label_base, label_path = resolve_label_path(rasters, LABEL_BASE)
    if not label_path:
        raise FileNotFoundError("Reference raster (Srbija pozari 400m grid_resampled.tif) not found.")

    ref_ds = gdal.Open(label_path)
    gt = ref_ds.GetGeoTransform()
    proj = ref_ds.GetProjection()
    rows, cols = ref_ds.RasterYSize, ref_ds.RasterXSize

    # load numeric sources (as float32)
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

    # ---- load model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = len(feat_order)
    model = KANModel(input_size=in_dim).to(device)
    state = torch.load(MODEL_WEIGHTS, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # ---- output raster ----
    drv = gdal.GetDriverByName("GTiff")
    if os.path.exists(OUT_TIF):
        drv.Delete(OUT_TIF)
    out_ds = drv.Create(OUT_TIF, cols, rows, 1, gdal.GDT_Float32, options=["COMPRESS=LZW","TILED=YES"])
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

    # ---- predict tile-wise with AMP ----
    use_amp = (device.type == "cuda")
    with torch.no_grad():
        for r0 in range(0, rows, TILE):
            r1 = min(rows, r0 + TILE); h = r1 - r0
            for c0 in range(0, cols, TILE):
                c1 = min(cols, c0 + TILE); w = c1 - c0

                valid = np.ones((h, w), dtype=bool)

                # numeric features (min-max scale)
                num_stack = []
                for name in num_cols:
                    tile = num_arrays[name][r0:r1, c0:c1]
                    nd = num_nodata[name]
                    if nd is not None:
                        valid &= (tile != nd)
                    tile = minmax_scale_inline(name, tile, minmax)
                    num_stack.append(tile)

                # categorical OHE per schema; skip clouds
                ohe_map = {}
                for base, cats in ohe_schema.items():
                    tile = cat_arrays[base][r0:r1, c0:c1]
                    nd = cat_nodata[base]
                    if nd is not None:
                        valid &= (tile != nd)
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

                # build feature tensor EXACT order: [num_cols..., ohe_cols...]
                X_tiles = []
                for name in num_cols:
                    X_tiles.append(num_stack[num_cols.index(name)])
                for col in ohe_cols:
                    base, val = col.split("==", 1)
                    k = int(val)
                    X_tiles.append(ohe_map[(base, k)])

                X_block = np.stack(X_tiles, axis=-1)   # (h, w, F)
                X_flat = X_block.reshape(-1, X_block.shape[-1])
                vmask = valid.reshape(-1)
                Xv = X_flat[vmask]
                if Xv.size == 0:
                    out_band.WriteArray(np.full((h, w), np.float32(OUT_NODATA)), xoff=c0, yoff=r0)
                    rgb_r.WriteArray(np.zeros((h, w), dtype=np.uint8), xoff=c0, yoff=r0)
                    rgb_g.WriteArray(np.zeros((h, w), dtype=np.uint8), xoff=c0, yoff=r0)
                    rgb_b.WriteArray(np.zeros((h, w), dtype=np.uint8), xoff=c0, yoff=r0)
                    continue

                Xt = torch.from_numpy(Xv).to(device)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits = model(Xt).squeeze(1)
                    probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)

                out_tile = np.full((h*w,), np.float32(OUT_NODATA), dtype=np.float32)
                out_tile[vmask] = probs
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
    print("✅ Wrote raster probabilities  →", os.path.abspath(OUT_TIF))
    print("🎨 Wrote RGB visualization     →", os.path.abspath(OUT_RGB))

if __name__ == "__main__":
    main()
