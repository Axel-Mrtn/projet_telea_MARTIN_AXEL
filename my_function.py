import os
from osgeo import gdal
import numpy as np

def rasterisation(in_vector, ref_image, out_image, field_name, dtype="Int32", nodata=0, all_touched=True):
    """
    Rasterise un vecteur (shp) sur la grille d'un raster de référence,
    en gravant la valeur du champ field_name.
    """

    ds = gdal.Open(ref_image, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Impossible d'ouvrir le raster de référence: {ref_image}")

    gt = ds.GetGeoTransform()
    xsize, ysize = ds.RasterXSize, ds.RasterYSize

    xres = gt[1]
    yres = abs(gt[5])

    xmin = gt[0]
    ymax = gt[3]
    xmax = xmin + xsize * xres
    ymin = ymax - ysize * yres

    ds = None


    os.makedirs(os.path.dirname(out_image), exist_ok=True)
    at_opt = "-at " if all_touched else ""

    cmd = (
        f"gdal_rasterize {at_opt}"
        f"-a {field_name} "
        f"-tr {xres} {yres} "
        f"-te {xmin} {ymin} {xmax} {ymax} "
        f"-a_nodata {nodata} -ot {dtype} -of GTiff "
        f"\"{in_vector}\" \"{out_image}\""
    )

    print(cmd)


    return os.system(cmd)




def compute_ari_timeseries(B3, B5, nodata=-9999):
    B3 = B3.astype("float32")
    B5 = B5.astype("float32")

    # Masque stricte pour éviter divisions par zéro et valeurs trop petites
    epsilon = 1e-6
    mask = (B3 > epsilon) & (B5 > epsilon)

    ari = np.full(B3.shape, nodata, dtype="float32")

    ari[mask] = (
        (1.0 / B3[mask] - 1.0 / B5[mask]) /
        (1.0 / B3[mask] + 1.0 / B5[mask])
    )

    print("ARI stats: min =", ari[mask].min(), "max =", ari[mask].max(), "mean =", ari[mask].mean())

    return ari




import os
import numpy as np
from osgeo import gdal

def build_multiband_stack_with_ari(folder, prefix, bands, ari_path, dtype=np.float32):
    """
    Construit un stack multi-bandes (Pyrénées) et le fusionne avec un raster Ari
    en concaténant les bandes (features).

    Returns
    -------
    image_stack_fused : np.ndarray (rows, cols, nb_bands_total)
    ds_ref : GDAL Dataset (référence géométrie)
    """

    # --- 1) stack Pyrénées (ton code)
    tif_list = [os.path.join(folder, f"{prefix}_{band}.tif") for band in bands]

    stack = []
    ds_ref = None
    n_times_ref = None

    for tif in tif_list:
        ds = gdal.Open(tif)
        if ds is None:
            raise FileNotFoundError(f"Impossible d'ouvrir {tif}")

        if ds_ref is None:
            ds_ref = ds

        arr = ds.ReadAsArray()  # (times, rows, cols) ou (rows, cols)
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]

        n_times = arr.shape[0]
        if n_times_ref is None:
            n_times_ref = n_times
        elif n_times != n_times_ref:
            raise ValueError("Incohérence temporelle entre bandes")

        arr = np.transpose(arr, (1, 2, 0))  # (rows, cols, times)
        stack.append(arr)

    image_stack_py = np.concatenate(stack, axis=2).astype(dtype)  # (rows, cols, times*nb_bands)

    # --- 2) lire Ari
    ds_ari = gdal.Open(ari_path)
    if ds_ari is None:
        raise FileNotFoundError(f"Impossible d'ouvrir Ari : {ari_path}")

    ari_arr = ds_ari.ReadAsArray()  # (bands, rows, cols) ou (rows, cols)
    if ari_arr.ndim == 2:
        ari_arr = ari_arr[np.newaxis, :, :]  # (1, rows, cols)
    ari_arr = np.transpose(ari_arr, (1, 2, 0)).astype(dtype)  # (rows, cols, nb_bands_ari)

    # --- 3) checks alignement
    if (ds_ari.RasterXSize != ds_ref.RasterXSize) or (ds_ari.RasterYSize != ds_ref.RasterYSize):
        raise ValueError("Ari n'a pas la même taille (rows/cols) que le stack Pyrénées. Il faut le reprojeter/resampler.")

    if ds_ari.GetGeoTransform() != ds_ref.GetGeoTransform():
        raise ValueError("Ari n'a pas le même GeoTransform que le stack Pyrénées. Il faut le reprojeter/resampler.")

    if ds_ari.GetProjection() != ds_ref.GetProjection():
        raise ValueError("Ari n'a pas la même projection que le stack Pyrénées. Il faut le reprojeter.")

    # --- 4) fusion (concaténation des features)
    image_stack_fused = np.concatenate([image_stack_py, ari_arr], axis=2).astype(dtype)

    return image_stack_fused, ds_ref
