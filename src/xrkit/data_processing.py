import os
import time
import math
import chardet
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
import rioxarray as rxr
import xarray as xr
import multiprocessing
from tqdm import tqdm
from affine import Affine
from pyproj import CRS
from geocube.api.core import make_geocube


def csv_to_shp(csv_path: str, shp_path: str,
               lon='lon', lat='lat',
               input_crs='epsg:4326',
               output_crs=None,
               encoding='auto'):
    """
    20250114
    Convert csv file to point shp file.

    :param csv_path: The path of the csv file.
    :param lon: The name of the longitude column.
    :param lat: The name of the latitude column.
    :param crs: The coordinate reference system.
    :param shp_path: The path of the shp file.
    """
    try:
        CRS.from_user_input(input_crs)
        CRS.from_user_input(output_crs) if output_crs else None
    except ValueError as e:
        raise ValueError(f"坐标系 {input_crs} 或 {output_crs} 是无效的。") from e

    # Detect the encoding of the csv file.
    if encoding == 'auto':
        f = open(csv_path, 'rb')
        data = f.read()
        encoding = chardet.detect(data).get('encoding')

    # Read the csv file and save it as a shapefile.
    df = gpd.read_file(csv_path, encoding=encoding)

    # Error check
    if lon not in df.columns or lat not in df.columns:
        raise ValueError(f"{lon} 或 {lat} 列不存在。")

    # Convert the lon and lat columns to numeric.
    df[[lon, lat]] = df[[lon, lat]].apply(pd.to_numeric)

    # Create a GeoDataFrame.
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon], df[lat]))

    # Set the CRS of the GeoDataFrame.
    gdf.crs = input_crs

    # If the output CRS is specified, convert the CRS.
    if output_crs:
        gdf = gdf.to_crs(output_crs)

    gdf.to_file(shp_path, driver='ESRI Shapefile', encoding=encoding)

    # Print the time when the file is saved.
    shp_name = os.path.basename(shp_path)
    print(f"{shp_name} saved ({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())})")


def zonal_statistics(value_file,
                     zone_file,
                     zone_field=None,
                     statistical_type='mean',
                     bins=None,
                     write_raster=None):
    """
    20250523
    Calculate zonal statistics.

    :param value_file: The path of the value raster (or xr.DataArray).
    :param zone_file: The path of the zone file (or gpd.GeoDataFrame).
    :param zone_field: The field name of the zone file.
    :param bins: The bins for grouping the zone data.
    :param statistical_type: The type of statistic.
    :param write_raster: The path of the output raster.
    """

    # Read the zone data and raster data
    if str(zone_file).endswith(".shp") or isinstance(zone_file, gpd.GeoDataFrame):
        if zone_field is None:
            raise ValueError("The field name 'zone_field' must be specified")

        if str(zone_file).endswith(".shp"):
            zone_data = gpd.read_file(zone_file)
        if isinstance(zone_file, gpd.GeoDataFrame):
            zone_data = zone_file
        if str(value_file).endswith(".tif"):
            value_data = rxr.open_rasterio(value_file, masked=True)
        if isinstance(value_file, xr.DataArray):
            value_data = value_file

        value_data = value_data.rio.clip(
            zone_data.geometry.values, zone_data.crs, from_disk=True
        ).sel(band=1).drop_vars("band").astype(float)

        # If the zone_field of the shp is text
        if zone_data[zone_field].dtype == 'object':
            # Convert zone_field to numeric
            zone_data['zone'] = pd.Categorical(zone_data[zone_field])
            zone_data['zone'] = zone_data['zone'].cat.codes
        else:
            zone_data['zone'] = zone_data[zone_field]

        # Rasterize the zone data
        combined_data = make_geocube(
            vector_data=zone_data,
            measurements=['zone'],
            like=value_data,
        )

        # Combine the zone data and value data
        combined_data["value"] = value_data

    # Read the raster data
    elif str(zone_file).endswith(".tif") or isinstance(zone_file, xr.DataArray):
        if str(zone_file).endswith(".tif"):
            zone_data = rxr.open_rasterio(zone_file, masked=True)
        if isinstance(zone_file, xr.DataArray):
            zone_data = zone_file
        if str(value_file).endswith(".tif"):
            value_data = rxr.open_rasterio(value_file, masked=True)
        if isinstance(value_file, xr.DataArray):
            value_data = value_file

        zone_data = zone_data.sel(band=1).drop_vars("band")
        value_data = value_data.sel(band=1).drop_vars("band")

        # Align the zone data and value data
        if zone_data.rio.resolution()[0] <= value_data.rio.resolution()[0]:
            projected_value_data = value_data.rio.reproject_match(
                zone_data,
                resampling=rio.enums.Resampling.nearest
            )
            combined_data = xr.merge(
                [zone_data.rename('zone'), projected_value_data.rename('value')])
        else:
            projected_zone_data = zone_data.rio.reproject_match(
                value_data,
                resampling=rio.enums.Resampling.nearest
            )
            combined_data = xr.merge(
                [projected_zone_data.rename('zone'), value_data.rename('value')])

    else:
        raise ValueError("zone_file must be a shp or tif file")

    # If bins is not None, group the zone data
    if bins is not None:
        zone_values = combined_data['zone'].values
        zone_values_na = np.isnan(zone_values)
        group_indices = np.digitize(zone_values, bins)
        combined_data['zone'].values[~zone_values_na] = group_indices[~zone_values_na]

    # Calculate zonal statistics
    combined_data = combined_data.set_coords('zone')
    grouped_value = combined_data.drop_vars("spatial_ref").groupby('zone')

    # Calculate statistics
    if statistical_type == 'mean':
        zonal_stat = grouped_value.mean().rename({"value": statistical_type})
    elif statistical_type == 'max':
        zonal_stat = grouped_value.max().rename({"value": statistical_type})
    elif statistical_type == 'min':
        zonal_stat = grouped_value.min().rename({"value": statistical_type})
    elif statistical_type == 'sum':
        zonal_stat = grouped_value.sum().rename({"value": statistical_type})
    elif statistical_type == 'median':
        zonal_stat = grouped_value.median().rename({"value": statistical_type})
    elif statistical_type == 'std':
        zonal_stat = grouped_value.std().rename({"value": statistical_type})
    elif statistical_type == 'count':
        zonal_stat = grouped_value.count().rename({"value": statistical_type})
    else:
        raise ValueError(
            "statistical_type must be 'mean', 'max', 'min', 'sum', 'median', 'std' or 'count'")

    # Convert zonal_stat to DataFrame
    zonal_stat = zonal_stat.to_dataframe()

    # If bins is not None, assign group names to the zone_field column
    if bins is not None:
        group_names = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
        group_names.insert(0, f"<{bins[0]}")
        group_names.append(f">{bins[-1]}")

        # Assign group_names to the zone_field column
        zonal_stat['zone_value'] = zonal_stat.index
        zonal_stat.reset_index(drop=True, inplace=True)
        zonal_stat['classes'] = ''
        for i in range(len(group_names)):
            zonal_stat.loc[zonal_stat['zone_value']
                           == i, 'classes'] = group_names[i]
    else:
        if zone_data[zone_field].dtype == 'object':
            # Map the zonal_stat.index to the zone_field column
            zonal_stat['classes'] = zonal_stat.index.map(
                zone_data.set_index('zone')[zone_field])
            zonal_stat.reset_index(drop=True, inplace=True)
        else:
            zonal_stat['classes'] = zonal_stat.index
            zonal_stat.reset_index(drop=True, inplace=True)

    # If write_raster is not None, export to write_raster
    if write_raster is not None:
        # Assign the results of zonal_stat to the tif file
        if bins is not None:
            for i in range(len(zonal_stat)):
                combined_data['zone'].values[combined_data['zone'].values ==
                                             zonal_stat.loc[i, 'zone_value']] = zonal_stat.loc[i, statistical_type]
        else:
            for i in range(len(zonal_stat)):
                combined_data['zone'].values[combined_data['zone'].values ==
                                             zonal_stat.loc[i, 'classes']] = zonal_stat.loc[i, statistical_type]

        # Export to write_raster
        combined_data['zone'].rio.to_raster(write_raster)
        print(f"{os.path.basename(write_raster)} saved ({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())})")

    # If zonal_stat has 'zone_value' column, drop it
    if 'zone_value' in zonal_stat.columns:
        zonal_stat.drop(columns='zone_value', inplace=True)

    return zonal_stat


def reproject_to_wgs84(input_raster,
                       output_raster,
                       resolution,
                       resample_method='n'):
    """
    20250311
    Reproject the input raster (tif path or xr.DataArray) to WGS84 (EPSG:4326).
    The output image only covers the input data area, but the pixels must align with the global grid.
    The center coordinates of the first (top-left) pixel of the global grid are (resolution/2, 180 - resolution/2).

    :param input_raster: The path of the input raster (or xr.DataArray).
    :param output_raster: The path of the output raster.
    :param resolution: The targeted resolution.
    :param resample_method: The resampling method.
    """

    # 1. If the input is a tif file path, open it with rioxarray
    if str(input_raster).endswith(".tif"):
        da = rxr.open_rasterio(input_raster)
    elif isinstance(input_raster, xr.DataArray):
        da = input_raster
    else:
        raise ValueError(
            "input_raster must be a tif path or an xr.DataArray object")

    # 2. The target coordinate system is EPSG:4326 (WGS84)
    target_crs = "EPSG:4326"

    # 3. Get the bounds of the input data in its original CRS and transform them to the target CRS
    src_crs = da.rio.crs
    left, bottom, right, top = da.rio.bounds()
    t_left, t_bottom, t_right, t_top = rio.warp.transform_bounds(
        src_crs, target_crs, left, bottom, right, top, densify_pts=21)

    # 4. Align the output extent with the global target grid
    #
    # The global target grid is defined (assuming the global grid's initial transform is):
    #   T_global = Affine(res, 0, 0, 0, -res, 180)
    # The pixel center is at (0 + res/2, 180 - res/2) for the first pixel,
    # and each subsequent column/row adds/subtracts res.
    #
    # For the x direction: pixel center x = res/2 + k·res, where k is an integer,
    # find the smallest integer k_min such that the center is not less than t_left,
    # and the largest integer k_max such that the center is not greater than t_right.
    k_min = math.ceil((t_left - resolution/2) / resolution)
    k_max = math.floor((t_right - resolution/2) / resolution)
    # For the y direction: pixel center y = 180 - res/2 - l·res, where l is an integer,
    # Note: In geographic coordinates, larger y values are at the top.
    # Find the smallest integer l_min such that the center is not less than t_top
    #  (t_top is the upper bound after reprojection),
    # and the largest integer l_max such that the center is not greater than t_bottom.
    l_min = math.ceil((180 - resolution/2 - t_top) / resolution)
    l_max = math.floor((180 - resolution/2 - t_bottom) / resolution)

    # The number of rows and columns in the output image
    out_width = k_max - k_min + 1
    out_height = l_max - l_min + 1

    if out_width <= 0 or out_height <= 0:
        raise ValueError(
            "The calculated output extent is empty, please check the input data and resolution settings.")

    # 5. Construct the affine transform for the output raster
    # For the global grid T_global = Affine(res, 0, 0, 0, -res, 180),
    # the top-left pixel of the subgrid is the (k_min, l_min) pixel in the global grid,
    # its center coordinates are (res/2 + k_min·res, 180 - res/2 - l_min·res).
    # Construct the affine transform for the subgrid:
    new_transform = Affine(resolution, 0, k_min * resolution,
                           0, -resolution, 180 - l_min * resolution)

    # 6. Choose the resampling method
    resample_dict = {
        'n': rio.enums.Resampling.nearest,
        'b': rio.enums.Resampling.bilinear,
        'c': rio.enums.Resampling.cubic,
        # Add more options as needed
    }
    resampling = resample_dict.get(resample_method.lower())

    # 7. Use rioxarray's reproject method to perform the reprojection,
    #    specifying the target CRS, output affine transform, output shape, and resampling method
    reprojected = da.rio.reproject(
        target_crs,
        transform=new_transform,
        shape=(out_height, out_width),
        resampling=resampling
    )

    # 8. Write the reprojected data to the specified path
    reprojected.rio.to_raster(output_raster)

    # 9. Print the time when the file is saved.
    output_tif_name = os.path.basename(output_raster)
    print(f"{output_tif_name} saved ({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())})")


def aggregate(input_raster,
              output_raster,
              resolution,
              output_crs='EPSG:4326',
              statistical_type='mean',
              resample_method='n',
              align=True):
    """
    20250311
    Perform upscaling on the input raster

    :param input_raster: path to the input raster (or xr.DataArray)
    :param output_raster: path to the output raster
    :param resolution: target resolution (unit: degree)
    :param output_crs: optional, output CRS or tif path, default is 'EPSG:4326'
    :param resample_method: str, optional, resample method, default is 'n' (nearest)
    :param statistical_type: str, optional, statistical type, default is 'mean'
    :param align: bool, optional, whether or not to align the raster when output_crs is a tif path
    """

    # Open the input raster
    if str(input_raster).endswith(".tif"):
        raster = rxr.open_rasterio(input_raster)
    elif isinstance(input_raster, xr.DataArray):
        raster = input_raster
    else:
        raise ValueError(
            "input_raster must be a tif path or an xr.DataArray object")

    # Choose the resampling method
    resample_dict = {
        'n': rio.enums.Resampling.nearest,
        'b': rio.enums.Resampling.bilinear,
        'c': rio.enums.Resampling.cubic,
        # Add more options as needed
    }
    resampling = resample_dict.get(resample_method.lower())

    # Reproject
    if str(output_crs).endswith(".tif"):
        reference_raster = rxr.open_rasterio(output_crs)
        reference_crs = reference_raster.rio.crs
        raster = raster.rio.reproject(
            dst_crs=reference_crs,
            resampling=rio.enums.Resampling.nearest
        )
    elif str(output_crs).lower().startswith("epsg"):
        raster = raster.rio.reproject(
            dst_crs=output_crs,
            resampling=resampling)
    else:
        raise ValueError(
            "output_crs必须为一个epsg格式的坐标系或是一个tif栅格。")

    # Calculate the resolution ratio of the new raster
    orig_res_x = orig_res_y = abs(raster.rio.resolution()[0])
    scale_factor_x = int(resolution / orig_res_x)
    scale_factor_y = int(resolution / orig_res_y)

    # Statistical methods mapping
    statistical_methods = {
        'mean': np.nanmean,
        'median': np.nanmedian,
        'max': np.nanmax,
        'min': np.nanmin,
        'sum': np.nansum,
        'std': np.nanstd,
    }
    if statistical_type in statistical_methods:
        new_data = raster.coarsen(
            x=scale_factor_x, y=scale_factor_y, boundary='trim').reduce(statistical_methods[statistical_type])
    elif statistical_type == 'count':
        new_data = raster.coarsen(
            x=scale_factor_x, y=scale_factor_y, boundary='trim').count()
    else:
        raise ValueError(f"Unknown statistical type: {statistical_type}")

    # Resample to ensure new_data has the target resolution
    if str(output_crs).endswith(".tif"):
        if align==True:
            new_data = new_data.rio.reproject_match(
                reference_raster,
                resampling=rio.enums.Resampling.nearest
            )
        elif align==False:
            new_data = new_data.rio.reproject(
                dst_crs=reference_crs,
                resolution=resolution,
                resampling=rio.enums.Resampling.nearest)
    elif str(output_crs).lower().startswith("epsg"):
        new_data = new_data.rio.reproject(
            dst_crs=output_crs,
            resolution=resolution,
            resampling=rio.enums.Resampling.nearest)
    else:
        raise ValueError(
            "output_crs必须为一个epsg格式的坐标系或是一个tif栅格。")

    # Update metadata and save
    new_data.rio.to_raster(output_raster)

    output_tif_name = os.path.basename(output_raster)
    print(f"{output_tif_name} saved ({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())})")


def parafun(fun, params, ncore=multiprocessing.cpu_count()):
    """
    20250311
    Run a function in parallel

    :param fun: function, the function to run
    :param params: list, the parameters for the function
    :param ncore: int, the number of cores to use
    :return: list, the results of the function
    """
    ncore = min(ncore, len(params))
    pool = multiprocessing.Pool(ncore)
    results = list(tqdm(pool.imap(fun, params), total=len(params)))
    pool.close()
    pool.join()
    
    print("All tasks are done.")
    return results