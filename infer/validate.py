import numpy as np
from typing import List
from scipy.optimize import least_squares

from infer.rs_image import RSImage


def compute_multiview_pair_errors(images: List[RSImage], height_fusion: str = "median", lm_max_nfev: int = 50):
    if len(images) < 2:
        return np.array([])

    tie_points_list = [image.tie_points for image in images]
    if any(tp is None for tp in tie_points_list):
        return np.array([])

    lengths = [tp.shape[0] for tp in tie_points_list]
    if len(set(lengths)) != 1:
        raise ValueError("All images must have the same number of tie points.")

    tie_points_stack = np.stack(tie_points_list, axis=0)
    lines_stack = tie_points_stack[:, :, 0]
    samps_stack = tie_points_stack[:, :, 1]

    dem_stack = []
    for image, lines, samps in zip(images, lines_stack, samps_stack):
        dem_stack.append(image.dem[lines, samps])
    dem_stack = np.stack(dem_stack, axis=0)

    if height_fusion == "median":
        Z_k = np.median(dem_stack, axis=0)
    elif height_fusion == "mean":
        Z_k = np.mean(dem_stack, axis=0)
    else:
        raise ValueError(f"Unsupported height_fusion: {height_fusion}")

    lat_init = []
    lon_init = []
    for image, lines, samps in zip(images, lines_stack, samps_stack):
        lats, lons = image.rpc.RPC_PHOTO2OBJ(samps, lines, Z_k, 'numpy')
        lat_init.append(lats)
        lon_init.append(lons)
    lat_init = np.stack(lat_init, axis=0)
    lon_init = np.stack(lon_init, axis=0)

    lat0 = np.nanmedian(lat_init, axis=0)
    lon0 = np.nanmedian(lon_init, axis=0)

    num_images, num_points = lines_stack.shape
    residuals_per_point = np.zeros((num_points, num_images), dtype=np.float64)

    for k in range(num_points):
        z_k = Z_k[k]
        lat0_k = lat0[k]
        lon0_k = lon0[k]

        def residual_func(x):
            lat, lon = x
            res = []
            for i in range(num_images):
                samps_pred, lines_pred = images[i].rpc.RPC_OBJ2PHOTO(lat, lon, z_k, 'numpy')
                res.append(lines_pred - lines_stack[i, k])
                res.append(samps_pred - samps_stack[i, k])
            return np.array(res, dtype=np.float64)

        result = least_squares(residual_func, x0=np.array([lat0_k, lon0_k], dtype=np.float64), method="lm", max_nfev=lm_max_nfev)
        lat_hat, lon_hat = result.x

        for i in range(num_images):
            samps_pred, lines_pred = images[i].rpc.RPC_OBJ2PHOTO(lat_hat, lon_hat, z_k, 'numpy')
            residuals_per_point[k, i] = np.sqrt((lines_pred - lines_stack[i, k]) ** 2 + (samps_pred - samps_stack[i, k]) ** 2)

    return residuals_per_point.reshape(-1)
