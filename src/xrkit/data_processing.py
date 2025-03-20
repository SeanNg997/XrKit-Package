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


def csv_to_shp(input_file: str, output_file: str,
               lon='lon', lat='lat',
               input_crs='epsg:4326',
               output_crs='epsg:4326',
               encoding='auto'):
    """
    Convert a csv file to a shp file.
    
    Parameters
    ----------
    input_file : str
        The path of the csv file.
    output_file : str
        The path of the shp file.
    lon : str, optional
        The name of the longitude column.
    lat : str, optional
        The name of the latitude column.
    input_crs : str, optional
        The crs of the points in the csv file, default is 'epsg:4326'.
    output_crs : str, optional
        The target crs of the shp file, default is 'epsg:4326'.
    encoding : str, optional
        The encoding of the csv file. Default is 'auto'.

    Returns
    -------
    none
    
    Updated
    -------
    20250320
    """

    # Detect the encoding of the csv file.
    if encoding == 'auto':
        f = open(input_file, 'rb')
        data = f.read()
        encoding = chardet.detect(data).get('encoding')

    # Read the csv file and save it as a shapefile.
    df = gpd.read_file(input_file, encoding=encoding)

    # Create a GeoDataFrame.
    if lon not in df.columns or lat not in df.columns:
        raise ValueError(f"{lon} 或 {lat} 列不存在。")
    df[[lon, lat]] = df[[lon, lat]].apply(pd.to_numeric)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon], df[lat]))
    gdf.crs = input_crs

    # If the output CRS is specified, convert the CRS.
    if output_crs != input_crs:
        gdf = gdf.to_crs(output_crs)

    # Save the GeoDataFrame as a shapefile.
    gdf.to_file(output_file, driver='ESRI Shapefile', encoding=encoding)

    # Print the time when the file is saved.
    shp_name = os.path.basename(output_file)
    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f"{shp_name} saved ({t})")


def zonal_statistics(value_file: str,
                     zone_file: str,
                     output_file: str,
                     zone_field=None,
                     bins=None,
                     statistical_type='mean'):
    """
    Calculate zonal statistics.

    Parameters
    ----------
    value_file : str
        The path of the value raster.
    zone_file : str
        The path of the zone file.
    output_file : str
        The path of the output raster.
    zone_field : str, optional
        The field name of the zone file, default is None.
    bins : list, optional
        The bins for grouping the zone data, default is None.
    statistical_type : str, optional
        The type of statistic, default is 'mean'.

    Returns
    -------
    DataFrame
        A DataFrame containing the zonal statistics.

    Updated
    -------
    20250523
    """

    # Error check for input parameters
    if not str(value_file).endswith(".tif"):
        raise ValueError("value_file must be a path to a tif file.")
    if not str(zone_file).endswith(".shp") and not str(zone_file).endswith(".tif"):
        raise ValueError("zone_file must be a path to a shp or tif file.")
    if not str(output_file).endswith(".tif"):
        raise ValueError("output_file must be a path to a tif file.")
    if str(zone_file).endswith(".shp") and zone_field is None:
        raise ValueError("The field name 'zone_field' must be specified")
    
    # Read the zone data and value data
    if str(zone_file).endswith(".shp"):
        zone_data = gpd.read_file(zone_file)
        value_data = rxr.open_rasterio(value_file, masked=True)
        
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
    elif str(zone_file).endswith(".tif"):
        zone_data = rxr.open_rasterio(zone_file, masked=True)
        value_data = rxr.open_rasterio(value_file, masked=True)

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
            zonal_stat['zone_value'] = zonal_stat.index
            zonal_stat['classes'] = zonal_stat.index.map(
                zone_data.set_index('zone')[zone_field])
            zonal_stat.reset_index(drop=True, inplace=True)
        else:
            zonal_stat['zone_value'] = zonal_stat.index
            zonal_stat['classes'] = zonal_stat.index
            zonal_stat.reset_index(drop=True, inplace=True)

    # Assign the results of zonal_stat to the tif file
    for i in range(len(zonal_stat)):
        combined_data['zone'].values[combined_data['zone'].values ==
                                        zonal_stat.loc[i, 'zone_value']] = zonal_stat.loc[i, statistical_type]

    # Export to output_file
    combined_data['zone'].rio.to_raster(output_file)
    file_name = os.path.basename(output_file)
    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f"{file_name} saved ({t})")

    # If zonal_stat has 'zone_value' column, drop it
    if 'zone_value' in zonal_stat.columns:
        zonal_stat.drop(columns='zone_value', inplace=True)

    # 把zonal_stat的classes列调整到第一列
    # Move the 'classes' column to the first position
    cols = ['classes'] + [col for col in zonal_stat.columns if col != 'classes']
    zonal_stat = zonal_stat[cols]
    return zonal_stat


def zonal_statistics_i(value_file: xr.DataArray,
                        zone_file,
                        zone_field=None,
                        bins=None,
                        statistical_type='mean'):
    """
    Calculate zonal statistics (interactive).

    Parameters
    ----------
    value_file : xr.DataArray
        A xr.DataArray object.
    zone_file : str
        TA xr.DataArray object or a gpd.GeoDataFrame object.
    zone_field : str, optional
        The field name of the zone file, default is None.
    bins : list, optional
        The bins for grouping the zone data, default is None.
    statistical_type : str, optional
        The type of statistic, default is 'mean'.

    Returns
    -------
    DataFrame
        A DataFrame containing the zonal statistics.
    xr.DataArray
        A xr.DataArray object containing the result data.
        
    Updated
    -------
    20250523
    """

    # Error check for input parameters
    if not isinstance(zone_file, gpd.GeoDataFrame) and not isinstance(zone_file, xr.DataArray):
        raise ValueError("zone_file must be a gpd.GeoDataFrame object or a xr.DataArray object.")
    if isinstance(zone_file, gpd.GeoDataFrame) and zone_field is None:
        raise ValueError("The field name 'zone_field' must be specified")

    # Read the zone data and raster data
    if isinstance(zone_file, gpd.GeoDataFrame):
        zone_data = zone_file

        value_data = value_file.rio.clip(
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
    elif isinstance(zone_file, xr.DataArray):
        zone_data = zone_file.sel(band=1).drop_vars("band")
        value_data = value_file.sel(band=1).drop_vars("band")

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
            zonal_stat['zone_value'] = zonal_stat.index
            zonal_stat['classes'] = zonal_stat.index.map(
                zone_data.set_index('zone')[zone_field])
            zonal_stat.reset_index(drop=True, inplace=True)
        else:
            zonal_stat['zone_value'] = zonal_stat.index
            zonal_stat['classes'] = zonal_stat.index
            zonal_stat.reset_index(drop=True, inplace=True)

    # Assign the results of zonal_stat to the tif file
    for i in range(len(zonal_stat)):
        combined_data['zone'].values[combined_data['zone'].values ==
                                        zonal_stat.loc[i, 'zone_value']] = zonal_stat.loc[i, statistical_type]

    # If zonal_stat has 'zone_value' column, drop it
    if 'zone_value' in zonal_stat.columns:
        zonal_stat.drop(columns='zone_value', inplace=True)
    cols = ['classes'] + [col for col in zonal_stat.columns if col != 'classes']
    zonal_stat = zonal_stat[cols]

    return zonal_stat, combined_data


def reproject_to_wgs84(input_file: str,
                       output_file: str,
                       resolution: float,
                       resample_method='n'):
    """
    Reproject the input raster to WGS84.

    Parameters
    ----------
    input_file : str
        The path of the input raster.
    output_file : str
        The path of the output raster.
    resolution : float
        The targeted resolution.
    resample_method : str, optional
        The resampling method, default is 'n' (nearest).

    Returns
    -------
    None

    Updated
    -------
    20250320
    """
    # error check
    if not str(input_file).endswith(".tif"):
        raise ValueError("input_file must be a path to a tif file.")
    if not str(output_file).endswith(".tif"):
        raise ValueError("output_file must be a path to a tif file.")
    if resolution <= 0:
        raise ValueError("resolution must be greater than 0.")
    if resample_method not in ['n', 'b', 'c']:
        raise ValueError("resample_method must be 'n'(nearest), 'b'(bilinear) or 'c'(cubic).")
    
    da = rxr.open_rasterio(input_file)
    src_crs = da.rio.crs
    target_crs = "EPSG:4326"
    left, bottom, right, top = da.rio.bounds()
    t_left, t_bottom, t_right, t_top = rio.warp.transform_bounds(
        src_crs, target_crs, left, bottom, right, top, densify_pts=21)

    # Align the output extent
    k_min = math.ceil((t_left - resolution/2) / resolution)
    k_max = math.floor((t_right - resolution/2) / resolution)
    l_min = math.ceil((180 - resolution/2 - t_top) / resolution)
    l_max = math.floor((180 - resolution/2 - t_bottom) / resolution)

    # The number of rows and columns in the output image
    out_width = k_max - k_min + 1
    out_height = l_max - l_min + 1

    if out_width <= 0 or out_height <= 0:
        raise ValueError(
            "The calculated output extent is empty, please check the input data and resolution settings.")

    # Construct the affine transform for the output raster
    new_transform = Affine(resolution, 0, k_min * resolution,
                           0, -resolution, 180 - l_min * resolution)

    # Choose the resampling method
    resample_dict = {
        'n': rio.enums.Resampling.nearest,
        'b': rio.enums.Resampling.bilinear,
        'c': rio.enums.Resampling.cubic,
        # Add more options as needed
    }
    resampling = resample_dict.get(resample_method.lower())

    # Perform the reprojection
    reprojected = da.rio.reproject(
        target_crs,
        transform=new_transform,
        shape=(out_height, out_width),
        resampling=resampling
    )

    # Write the reprojected data to the specified path
    reprojected.rio.to_raster(output_file, compress='LZW')

    # Print the save time.
    output_tif_name = os.path.basename(output_file)
    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f"{output_tif_name} saved ({t})")


def reproject_to_wgs84_i(input_file: xr.DataArray,
                         resolution: float,
                         resample_method='n'):
    """
    Reproject the input raster to WGS84 (interactive).

    Parameters
    ----------
    input_file : xr.DataArray
        The input raster.
    resolution : float
        The targeted resolution.
    resample_method : str, optional
        The resampling method, default is 'n' (nearest).

    Returns
    -------
    xr.DataArray
        The reprojected raster.

    Updated
    -------
    20250320
    """
    # error check
    if resolution <= 0:
        raise ValueError("resolution must be greater than 0.")
    if resample_method not in ['n', 'b', 'c']:
        raise ValueError("resample_method must be 'n'(nearest), 'b'(bilinear) or 'c'(cubic).")
    
    da = input_file
    src_crs = da.rio.crs
    target_crs = "EPSG:4326"
    left, bottom, right, top = da.rio.bounds()
    t_left, t_bottom, t_right, t_top = rio.warp.transform_bounds(
        src_crs, target_crs, left, bottom, right, top, densify_pts=21)

    # Align the output extent
    k_min = math.ceil((t_left - resolution/2) / resolution)
    k_max = math.floor((t_right - resolution/2) / resolution)
    l_min = math.ceil((180 - resolution/2 - t_top) / resolution)
    l_max = math.floor((180 - resolution/2 - t_bottom) / resolution)

    # The number of rows and columns in the output image
    out_width = k_max - k_min + 1
    out_height = l_max - l_min + 1

    if out_width <= 0 or out_height <= 0:
        raise ValueError(
            "The calculated output extent is empty, please check the input data and resolution settings.")

    # Construct the affine transform for the output raster
    new_transform = Affine(resolution, 0, k_min * resolution,
                           0, -resolution, 180 - l_min * resolution)

    # Choose the resampling method
    resample_dict = {
        'n': rio.enums.Resampling.nearest,
        'b': rio.enums.Resampling.bilinear,
        'c': rio.enums.Resampling.cubic,
        # Add more options as needed
    }
    resampling = resample_dict.get(resample_method.lower())

    # Perform the reprojection
    reprojected = da.rio.reproject(
        target_crs,
        transform=new_transform,
        shape=(out_height, out_width),
        resampling=resampling
    )

    return reprojected

def aggregate(input_file: str,
              output_file: str,
              resolution: float,
              output_crs='EPSG:4326',
              statistical_type='mean',
              resample_method='n'):
    """
    Perform aggregation on the input raster.

    Parameters
    ----------
    input_file : str
        Path to the input raster.
    output_file : str
        Path to the output raster.
    resolution : float
        Target resolution.
    output_crs : str, optional
        Output CRS or path to a reference raster, default is 'EPSG:4326'.
    resample_method : str, optional
        Resampling method, default is 'n' (nearest).
    statistical_type : str, optional
        Statistical type, default is 'mean'.

    Returns
    -------
    None

    Updated
    -------
    20250320
    """

    # Error check
    if not str(input_file).endswith(".tif"):
        raise ValueError("input_file must be a path to a tif file.")
    if not str(output_file).endswith(".tif"):
        raise ValueError("output_file must be a path to a tif file.")
    if resolution <= 0:
        raise ValueError("resolution must be greater than 0.")
    if resample_method not in ['n', 'b', 'c']:
        raise ValueError("resample_method must be 'n'(nearest), 'b'(bilinear) or 'c'(cubic).")
    if statistical_type not in ['mean', 'max', 'min', 'sum', 'median', 'std', 'count']:
        raise ValueError("statistical_type must be 'mean', 'max', 'min', 'sum', 'median', 'std' or 'count'.")
    
    raster = rxr.open_rasterio(input_file)

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
        new_data = new_data.rio.reproject_match(
            reference_raster,
            resampling=rio.enums.Resampling.nearest
        )
    elif str(output_crs).lower().startswith("epsg"):
        new_data = new_data.rio.reproject(
            dst_crs=output_crs,
            resolution=resolution,
            resampling=rio.enums.Resampling.nearest)
    else:
        raise ValueError(
            "output_crs必须为一个epsg格式的坐标系或是一个tif栅格。")

    # Update metadata and save
    new_data.rio.to_raster(output_file, compress='LZW')

    output_tif_name = os.path.basename(output_file)
    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f"{output_tif_name} saved ({t})")


def aggregate_i(input_file: xr.DataArray,
                resolution: float,
                output_crs='EPSG:4326',
                statistical_type='mean',
                resample_method='n'):
    """
    Perform aggregation on the input raster (interactive).

    Parameters
    ----------
    input_file : xr.DataArray
        The input raster.
    resolution : float
        Target resolution.
    output_crs : str, optional
        Output CRS or path to a reference raster, default is 'EPSG:4326'.
    resample_method : str, optional
        Resampling method, default is 'n' (nearest).
    statistical_type : str, optional
        Statistical type, default is 'mean'.

    Returns
    -------
    xr.DataArray
        The aggregated raster.

    Updated
    -------
    20250320
    """

    # Error check
    if resolution <= 0:
        raise ValueError("resolution must be greater than 0.")
    if resample_method not in ['n', 'b', 'c']:
        raise ValueError("resample_method must be 'n'(nearest), 'b'(bilinear) or 'c'(cubic).")
    if statistical_type not in ['mean', 'max', 'min', 'sum', 'median', 'std', 'count']:
        raise ValueError("statistical_type must be 'mean', 'max', 'min', 'sum', 'median', 'std' or 'count'.")
    
    raster = input_file

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
        new_data = new_data.rio.reproject_match(
            reference_raster,
            resampling=rio.enums.Resampling.nearest
        )
    elif str(output_crs).lower().startswith("epsg"):
        new_data = new_data.rio.reproject(
            dst_crs=output_crs,
            resolution=resolution,
            resampling=rio.enums.Resampling.nearest)
    else:
        raise ValueError(
            "output_crs必须为一个epsg格式的坐标系或是一个tif栅格。")

    # return 
    return new_data

def parafun(fun,
             params,
               ncore=multiprocessing.cpu_count()):
    """
    Run a function in parallel.

    Parameters
    ----------
    fun : function
        The function to run.
    params : list
        The parameters for the function.
    ncore : int, optional
        The number of cores to use, default is the maximum available.

    Returns
    -------
    list
        The results of the function.

    Updated
    -------
    20250320
    """

    ncore = min(ncore, len(params))
    pool = multiprocessing.Pool(ncore)
    results = list(tqdm(pool.imap(fun, params), total=len(params)))
    pool.close()
    pool.join()
    
    print("All tasks are done.")
    return results