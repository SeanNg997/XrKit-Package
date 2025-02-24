import os
import time
import chardet
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
import rioxarray as rxr
import xarray as xr

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
                    statistic_type='mean',
                    bins=None,
                    write_raster=None):
    """
    20250523
    Calculate zonal statistics.

    :param value_file: The path of the value raster.
    :param zone_file: The path of the zone file.
    :param zone_field: The field name of the zone file.
    :param bins: The bins for grouping the zone data.
    :param statistic_type: The type of statistic.
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
            combined_data = xr.merge([zone_data.rename('zone'), projected_value_data.rename('value')])
        else:
            projected_zone_data = zone_data.rio.reproject_match(
                value_data,
                resampling=rio.enums.Resampling.nearest
            )
            combined_data = xr.merge([projected_zone_data.rename('zone'), value_data.rename('value')])
            
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
    if statistic_type == 'mean':
        zonal_stat = grouped_value.mean().rename({"value": statistic_type})
    elif statistic_type == 'max':
        zonal_stat = grouped_value.max().rename({"value": statistic_type})
    elif statistic_type == 'min':
        zonal_stat = grouped_value.min().rename({"value": statistic_type})
    elif statistic_type == 'sum':
        zonal_stat = grouped_value.sum().rename({"value": statistic_type})
    elif statistic_type == 'median':
        zonal_stat = grouped_value.median().rename({"value": statistic_type})
    elif statistic_type == 'std':
        zonal_stat = grouped_value.std().rename({"value": statistic_type})
    elif statistic_type == 'count':
        zonal_stat = grouped_value.count().rename({"value": statistic_type})
    else:
        raise ValueError("statistic_type must be 'mean', 'max', 'min', 'sum', 'median', 'std' or 'count'")

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
            zonal_stat.loc[zonal_stat['zone_value'] == i, 'classes'] = group_names[i]
    else:
        if zone_data[zone_field].dtype == 'object':
            # Map the zonal_stat.index to the zone_field column
            zonal_stat['classes'] = zonal_stat.index.map(zone_data.set_index('zone')[zone_field])
            zonal_stat.reset_index(drop=True, inplace=True)
        else:
            zonal_stat['classes'] = zonal_stat.index
            zonal_stat.reset_index(drop=True, inplace=True)

    # If write_raster is not None, export to write_raster
    if write_raster is not None:
        # Assign the results of zonal_stat to the tif file
        if bins is not None:
            for i in range(len(zonal_stat)):
                combined_data['zone'].values[combined_data['zone'].values == zonal_stat.loc[i, 'zone_value']] = zonal_stat.loc[i, statistic_type]
        else:
            for i in range(len(zonal_stat)):
                combined_data['zone'].values[combined_data['zone'].values == zonal_stat.loc[i, 'classes']] = zonal_stat.loc[i, statistic_type]
        
        # Export to write_raster
        combined_data['zone'].rio.to_raster(write_raster)
        print(f"{os.path.basename(write_raster)} saved ({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())})")
    
    # If zonal_stat has 'zone_value' column, drop it
    if 'zone_value' in zonal_stat.columns:
        zonal_stat.drop(columns='zone_value', inplace=True)

    return zonal_stat