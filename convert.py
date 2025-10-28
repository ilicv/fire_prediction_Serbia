# tif_to_png_firemask.py
import numpy as np
from osgeo import gdal
from PIL import Image

INPUT_TIF = "Srbija pozari 400m grid_resampled.tif"
OUTPUT_PNG = "Srbija_pozari_mask.png"

# prag za "pozar": > THRESHOLD -> vatreni piksel
THRESHOLD = 0.0   # ako su vrednosti 0/1, ovo je ok; po želji stavi 0.5

def main():
    ds = gdal.Open(INPUT_TIF)
    if ds is None:
        raise FileNotFoundError(INPUT_TIF)

    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    nodata = band.GetNoDataValue()
    dtype = arr.dtype

    # kratka dijagnostika
    finite = np.isfinite(arr)
    arr_valid = arr[finite]
    print("Raster:", INPUT_TIF)
    print("  shape:", arr.shape, "| dtype:", dtype)
    print("  NoData:", nodata)
    if arr_valid.size > 0:
        print("  min/max:", float(np.nanmin(arr_valid)), float(np.nanmax(arr_valid)))
        print("  count(arr > 0):", int(np.sum(arr_valid > 0)))
    else:
        print("  (warning) nema validnih vrednosti?")

    # maska validnosti (isključi NoData ako je definisan)
    valid_mask = np.ones_like(arr, dtype=bool)
    if nodata is not None:
        valid_mask &= (arr != nodata)

    # POZAR = vrednosti > THRESHOLD, ali samo na valid pikselima
    fire_mask = (arr > THRESHOLD) & valid_mask

    # napravi belu RGB sliku
    h, w = arr.shape
    rgb = np.ones((h, w, 3), dtype=np.uint8) * 255

    # oboji POZAR crveno
    rgb[fire_mask] = [255, 0, 0]

    # sve ostalo (uklj. NoData) ostaje belo
    img = Image.fromarray(rgb, mode="RGB")
    img.save(OUTPUT_PNG)
    print("✅ PNG sačuvan →", OUTPUT_PNG)
    print("   crvenih piksela:", int(fire_mask.sum()))
    print("   belih piksela  :", int(h*w - fire_mask.sum()))

if __name__ == "__main__":
    main()
