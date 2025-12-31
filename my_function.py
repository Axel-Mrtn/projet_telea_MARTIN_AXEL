import os
from osgeo import gdal


def rasterization(in_vector, ref_image, out_image, field_name, dtype=None):
    """
    Rasterise un vecteur (shp) sur la grille d'un raster de référence,
    en gravant la valeur du champ field_name.
    """
    print("in_vector:", in_vector)
    print("ref_image:", ref_image)
    print("out_image:", out_image)
    print("field_name:", field_name)


# for those parameters, you know how to get theses information if you had to
sptial_resolution = 0.5
xmin = 748231.0
ymin = 6273800.0
xmax = 751231.0
ymax = 6276800.0

# define command pattern to fill with paremeters
cmd_pattern = ("gdal_rasterize -a {field_name} "
               "-tr {sptial_resolution} {sptial_resolution} "
               "-te {xmin} {ymin} {xmax} {ymax} -ot Byte -of GTiff "
               "{in_vector} {out_image}")

# fill the string with the parameter thanks to format function
cmd = cmd_pattern.format(in_vector=in_vector, xmin=xmin, ymin=ymin, xmax=xmax,
                         ymax=ymax, out_image=out_image, field_name=field_name,
                         sptial_resolution=sptial_resolution)

# execute the command in the terminal
os.system(cmd)