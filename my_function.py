import os
from osgeo import gdal

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
