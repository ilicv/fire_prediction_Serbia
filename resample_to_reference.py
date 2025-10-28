# resample_to_reference_with_firegrid.py
import os
from osgeo import gdal, gdal_array
import numpy as np

# === Configuration ===
reference_path = "Normalized burn ratio.tif"  # reference grid

# Numeric rasters (continuous values)
numeric_rasters = [
    "Air temperature.tif",
    "Aridity index.tif",
    "Consecutive dry days.tif",
    "Consecutive wet days.tif",
    "Distance from roads.tif",
    "Distance from settlements.tif",
    "Distance from water surfaces.tif",
    "Elevation.tif",
    "Global horizontal irradiance.tif",
    "Normalized burn ratio.tif",  # reference also listed here but will be skipped
    "Precipitation.tif",
    "Slope.tif",
    "Topographic wetness index.tif",
    "Wind exposition index.tif",
]

# Categorical rasters (discrete classes) - NEAREST neighbor
categorical_rasters = [
    "Aspect.tif",
    "Land use.tif",
    "Srbija pozari 400m grid.tif",  # << added fire grid (label)
]

all_rasters = numeric_rasters + categorical_rasters

# Output folder
out_dir = "resampled"
os.makedirs(out_dir, exist_ok=True)

# === Load Reference Info ===
ref_ds = gdal.Open(reference_path)
if ref_ds is None:
    raise FileNotFoundError(f"Reference raster not found: {reference_path}")

ref_proj = ref_ds.GetProjection()
ref_gt = ref_ds.GetGeoTransform()
ref_cols = ref_ds.RasterXSize
ref_rows = ref_ds.RasterYSize

xmin = ref_gt[0]
ymax = ref_gt[3]
xres = ref_gt[1]
yres = abs(ref_gt[5])
xmax = xmin + ref_cols * xres
ymin = ymax - ref_rows * yres
output_bounds = (xmin, ymin, xmax, ymax)

print(f"\n?? Using {reference_path} as reference:")
print(f"   Resolution : {xres} x {yres}")
print(f"   Bounds     : {output_bounds}")
print(f"   Size       : {ref_cols} cols x {ref_rows} rows\n")

# === Quick existence check ===
missing = [p for p in all_rasters if not os.path.exists(p)]
if missing:
    print("?? Missing files (will be skipped):")
    for m in missing:
        print("   -", m)
    print()

# === Helper: clean 0/1 without NoData for fire grid ===
def force_binary_no_nodata(path_tif):
    ds = gdal.Open(path_tif, gdal.GA_Update)
    if ds is None:
        print(f"? Could not open for post-fix: {path_tif}")
        return
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    nodata = band.GetNoDataValue()

    # 1) map all nonzero to 1, everything else to 0
    out = np.zeros_like(arr, dtype=np.uint8)
    out[arr != 0] = 1  # if source had 1/NoData or any stray values

    band.WriteArray(out)
    # 2) ensure **no** NoData on the band (we want explicit 0s)
    try:
        band.DeleteNoDataValue()
    except Exception:
        band.SetNoDataValue(None)
    band.FlushCache()
    ds = None
    print(f"   ? Post-fixed to clean 0/1 with no NoData: {path_tif}")

# === Warp All Rasters ===
for raster_path in all_rasters:
    if raster_path == reference_path:
        continue  # skip resampling the reference itself
    if not os.path.exists(raster_path):
        continue  # skip missing

    is_numeric = raster_path in numeric_rasters
    resample_method = "bilinear" if is_numeric else "near"

    base = os.path.splitext(os.path.basename(raster_path))[0]
    output_path = os.path.join(out_dir, base + "_resampled.tif")

    print(f"?? Resampling: {raster_path}")
    print(f"    ? Method: {resample_method}")
    print(f"    ? Output: {output_path}")

    # For the fire grid, make sure we get Byte type and background 0
    warp_kwargs = dict(
        destNameOrDestDS=output_path,
        srcDSOrSrcDSTab=raster_path,
        format="GTiff",
        outputBounds=output_bounds,
        xRes=xres,
        yRes=yres,
        dstSRS=ref_proj,
        resampleAlg=resample_method,
        multithread=True
    )

    if raster_path == "Srbija pozari 400m grid.tif":
        # nearest + Byte + explicit 0 background
        warp_kwargs.update({
            "dstNodata": 0,
            "outputType": gdal.GDT_Byte,
        })

    gdal.Warp(**warp_kwargs)

    # Post-fix fire grid to strict 0/1 with no NoData
    if raster_path == "Srbija pozari 400m grid.tif":
        force_binary_no_nodata(output_path)

print("\n? Resampling complete. Outputs are in:", os.path.abspath(out_dir))
