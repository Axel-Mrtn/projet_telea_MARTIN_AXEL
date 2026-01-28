import os
from osgeo import gdal
import numpy as np

def rasterisation(in_vector, ref_image, out_image, field_name, dtype="Int32", nodata=0, all_touched=True):
    """
    Rasterise un fichier vecteur sur la grille d’un raster de référence
    en utilisant un champ attributaire comme valeur de pixel.

Parameters
----------
in_vector : str
    Chemin vers le fichier vecteur d’entrée (shapefile) à rasteriser.
ref_image : str
    Chemin vers le raster de référence définissant la grille,
    la résolution et l’emprise spatiale.
out_image : str
    Chemin vers le fichier raster de sortie.
field_name : str
    Nom du champ attributaire utilisé pour affecter les valeurs aux pixels.
dtype : str, optional
    Type de données du raster de sortie (par défaut "Int32").
nodata : int or float, optional
    Valeur NoData attribuée aux pixels sans information (par défaut 0).
all_touched : bool, optional
    Si True, tous les pixels touchant un polygone sont rasterisés.
    Si False, seuls les pixels dont le centre est inclus dans le polygone sont pris en compte.
----------

  """

    # Ouverture du raster de référence en lecture seule
    ds = gdal.Open(ref_image, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Impossible d'ouvrir le raster de référence: {ref_image}")

    # Récupération de la transformation géographique et des dimensions du raster
    gt = ds.GetGeoTransform()
    xsize, ysize = ds.RasterXSize, ds.RasterYSize

    # Résolution spatiale du raster (taille du pixel)
    xres = gt[1]
    yres = abs(gt[5])

    # Calcul de l’emprise spatiale du raster de référence
    xmin = gt[0]
    ymax = gt[3]
    xmax = xmin + xsize * xres
    ymin = ymax - ysize * yres

    # Fermeture du raster de référence
    ds = None

    # Création du dossier de sortie si nécessaire
    os.makedirs(os.path.dirname(out_image), exist_ok=True)

    # Option all_touched : tous les pixels touchant un polygone sont affectés
    at_opt = "-at " if all_touched else ""

    # Construction de la commande gdal_rasterize
    cmd = (
        f"gdal_rasterize {at_opt}"
        f"-a {field_name} "
        f"-tr {xres} {yres} "
        f"-te {xmin} {ymin} {xmax} {ymax} "
        f"-a_nodata {nodata} -ot {dtype} -of GTiff "
        f"\"{in_vector}\" \"{out_image}\""
    )

    # Affichage de la commande exécutée (utile pour le débogage)
    print(cmd)

    # Exécution de la commande système de rasterisation
    return os.system(cmd)





def compute_ari_timeseries(B3, B5, nodata=-9999):
    """
    Calcule l’indice de réflectance des anthocyanes (ARI) pour un jeu de données
    multi-temporel à partir des bandes spectrales Sentinel-2.

    Parameters
    ----------
    B3 : ndarray
        Tableau contenant les valeurs de la bande verte (B03). Le tableau peut
        être multidimensionnel afin de représenter une série temporelle.
    B5 : ndarray
        Tableau contenant les valeurs de la bande red-edge (B05). Doit avoir
        la même forme que B3.
    nodata : int or float, optional
        Valeur utilisée pour remplir les pixels pour lesquels l’ARI ne peut
        pas être calculé (par défaut -9999).

    Returns
    -------
    ari : ndarray
        Tableau contenant les valeurs de l’ARI calculées pixel par pixel pour
        l’ensemble de la série temporelle. Les pixels pour lesquels l’indice
        ne peut pas être calculé sont affectés à la valeur nodata.
    """

    # Conversion des bandes B03 et B05 en type float pour garantir la précision
    # des calculs et éviter les erreurs liées aux divisions
    B3 = B3.astype("float32")
    B5 = B5.astype("float32")

    # Définition d’un seuil minimal pour éviter les divisions par zéro
    # et les valeurs numériques trop faibles
    epsilon = 1e-6
    mask = (B3 > epsilon) & (B5 > epsilon)

    # Initialisation du tableau ARI avec la valeur NoData
    ari = np.full(B3.shape, nodata, dtype="float32")

    # Calcul de l’indice ARI uniquement sur les pixels valides
    ari[mask] = (
        (1.0 / B3[mask] - 1.0 / B5[mask]) /
        (1.0 / B3[mask] + 1.0 / B5[mask])
    )

    # Affichage de statistiques descriptives sur les valeurs ARI calculées
    # (minimum, maximum et moyenne) pour contrôle et diagnostic
    print(
        "ARI stats: min =", ari[mask].min(),
        "max =", ari[mask].max(),
        "mean =", ari[mask].mean()
    )

    # Retour du raster ARI (série temporelle)
    return ari





def create_feature_stack(folder, prefix, bands, ari_path, dtype=np.float32):
    """
    Construit un stack multi-bandes à partir des bandes Sentinel-2
    et le fusionne avec un raster ARI en concaténant l’ensemble des bandes
    sous forme de variables explicatives (features).
    
    Returns
    -------
    image_stack_fused : np.ndarray
        Tableau de dimension (rows, cols, nb_bands_total) contenant
        l’ensemble des bandes spectrales et l’indice ARI.
    ds_ref : GDAL Dataset
        Dataset de référence contenant la géométrie et la projection.
    """

    # --- 1) Construction du stack multi-temporel des bandes Sentinel-2
    # Création de la liste des chemins vers les rasters des bandes
    tif_list = [os.path.join(folder, f"{prefix}_{band}.tif") for band in bands]

    stack = []
    ds_ref = None
    n_times_ref = None

    # Lecture successive de chaque bande spectrale
    for tif in tif_list:
        ds = gdal.Open(tif)
        if ds is None:
            raise FileNotFoundError(f"Impossible d'ouvrir {tif}")

        # Définition du premier raster comme référence géométrique
        if ds_ref is None:
            ds_ref = ds

        # Lecture des données raster
        arr = ds.ReadAsArray()  # (times, rows, cols) ou (rows, cols)
        if arr.ndim == 2:
            # Ajout d’une dimension temporelle si le raster n’est pas multi-temporel
            arr = arr[np.newaxis, :, :]

        # Vérification de la cohérence temporelle entre bandes
        n_times = arr.shape[0]
        if n_times_ref is None:
            n_times_ref = n_times
        elif n_times != n_times_ref:
            raise ValueError("Incohérence temporelle entre bandes")

        # Réorganisation des dimensions pour obtenir (rows, cols, times)
        arr = np.transpose(arr, (1, 2, 0))
        stack.append(arr)

    # Concaténation de toutes les bandes Sentinel-2 dans un seul stack
    image_stack_py = np.concatenate(stack, axis=2).astype(dtype)

    # --- 2) Lecture du raster ARI
    ds_ari = gdal.Open(ari_path)
    if ds_ari is None:
        raise FileNotFoundError(f"Impossible d'ouvrir Ari : {ari_path}")

    # Lecture des données ARI
    ari_arr = ds_ari.ReadAsArray()  # (bands, rows, cols) ou (rows, cols)
    if ari_arr.ndim == 2:
        # Ajout d’une dimension si l’ARI ne contient qu’une seule bande
        ari_arr = ari_arr[np.newaxis, :, :]

    # Réorganisation des dimensions pour être compatible avec le stack Sentinel-2
    ari_arr = np.transpose(ari_arr, (1, 2, 0)).astype(dtype)

    # --- 3) Vérifications d’alignement spatial
    # Vérification des dimensions spatiales
    if (ds_ari.RasterXSize != ds_ref.RasterXSize) or (ds_ari.RasterYSize != ds_ref.RasterYSize):
        raise ValueError("Ari n'a pas la même taille (rows/cols) que le stack Pyrénées. Il faut le reprojeter/resampler.")

    # Vérification de la géotransformation
    if ds_ari.GetGeoTransform() != ds_ref.GetGeoTransform():
        raise ValueError("Ari n'a pas le même GeoTransform que le stack Pyrénées. Il faut le reprojeter/resampler.")

    # Vérification de la projection
    if ds_ari.GetProjection() != ds_ref.GetProjection():
        raise ValueError("Ari n'a pas la même projection que le stack Pyrénées. Il faut le reprojeter.")

    # --- 4) Fusion finale des bandes Sentinel-2 et de l’ARI
    image_stack_fused = np.concatenate([image_stack_py, ari_arr], axis=2).astype(dtype)

    # Retour du stack fusionné et du raster de référence
    return image_stack_fused, ds_ref

