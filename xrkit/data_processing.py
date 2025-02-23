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


def zonal_statistics(value_raster: str,
                    zone_file: str,
                    zone_field=None,
                    statistic_type='mean',
                    bins=None,
                    write_raster=None):
    """
    20250523
    Calculate zonal statistics.

    :param value_raster: The path of the value raster.
    :param zone_file: The path of the zone file.
    :param zone_field: The field name of the zone file.
    :param bins: The bins for grouping the zone data.
    :param statistic_type: The type of statistic.
    :param write_raster: The path of the output raster.
    """
    
    # 当zone_file是shp文件时，必须指定分区的字段名zone_field
    if zone_file.endswith(".shp") and zone_field is None:
        raise ValueError("当“zone_field”是shp文件时，必须指定分区的字段名“zone_field”")

    # 读取分区数据和栅格数据
    if zone_file.endswith(".shp"):
        try:
            zone_data = gpd.read_file(zone_file)
            value_data = rxr.open_rasterio(value_raster, masked=True).rio.clip(
                zone_data.geometry.values, zone_data.crs, from_disk=True
            ).sel(band=1).drop_vars("band").astype(float)
        except:
            raise ValueError("读取zone_file或value_raster失败")

        # 如果shp的zone_field为文字
        if zone_data[zone_field].dtype == 'object':
            # 将zone_field转换为数字
            zone_data['zone'] = pd.Categorical(zone_data[zone_field])
            zone_data['zone'] = zone_data['zone'].cat.codes
        else:
            zone_data['zone'] = zone_data[zone_field]

        # 将分区数据栅格化
        combined_data = make_geocube(
            vector_data=zone_data,
            measurements=['zone'],
            like=value_data,
        )

        # 将分区数据和值数据合并
        combined_data["value"] = value_data

    # 读取栅格数据
    elif zone_file.endswith(".tif"):
        try:
            zone_data = rxr.open_rasterio(zone_file, masked=True).sel(band=1).drop_vars("band")
            value_data = rxr.open_rasterio(value_raster, masked=True).sel(band=1).drop_vars("band")
        except:
            raise ValueError("读取zone_file或value_raster失败")
        
        # 对齐分区数据和值数据
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
        raise ValueError("zone_file必须是shp或tif文件")

    # 若bins不为空，则对分区数据进行分组
    if bins is not None:
        zone_values = combined_data['zone'].values
        zone_values_na = np.isnan(zone_values)
        group_indices = np.digitize(zone_values, bins)
        combined_data['zone'].values[~zone_values_na] = group_indices[~zone_values_na]

    # 计算分区统计值
    combined_data = combined_data.set_coords('zone')
    grouped_value = combined_data.drop_vars("spatial_ref").groupby('zone')

    ## 计算统计值
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
        raise ValueError("统计类型必须是'mean', 'max', 'min', 'sum', 'median', 'std'或'count'")

    zonal_stat = zonal_stat.to_dataframe()
    
    # 若bins不为空，则将分组名称赋值到zone_field列
    if bins is not None:
        group_names = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
        group_names.insert(0, f"<{bins[0]}")
        group_names.append(f">{bins[-1]}")

        # 删把group_names赋值到zone_field列
        zonal_stat['zone_value'] = zonal_stat.index
        zonal_stat.reset_index(drop=True, inplace=True)
        zonal_stat['classes'] = ''
        for i in range(len(group_names)):
            zonal_stat.loc[zonal_stat['zone_value'] == i, 'classes'] = group_names[i]
    else:
        if zone_data[zone_field].dtype == 'object':
            # 通过查表将zonal_stat的zonal_stat.index转化为zone_field列赋值到zone_field列
            zonal_stat['classes'] = zonal_stat.index.map(zone_data.set_index('zone')[zone_field])
            zonal_stat.reset_index(drop=True, inplace=True)
        else:
            zonal_stat['classes'] = zonal_stat.index
            zonal_stat.reset_index(drop=True, inplace=True)

    # 若write_raster不为空，则导出到write_raster
    if write_raster is not None:
        # 根据zonal_stat的结果赋值到tif文件
        if bins is not None:
            for i in range(len(zonal_stat)):
                combined_data['zone'].values[combined_data['zone'].values == zonal_stat.loc[i, 'zone_value']] = zonal_stat.loc[i, statistic_type]
        else:
            for i in range(len(zonal_stat)):
                combined_data['zone'].values[combined_data['zone'].values == zonal_stat.loc[i, 'classes']] = zonal_stat.loc[i, statistic_type]
        
        # 导出到write_raster
        combined_data['zone'].rio.to_raster(write_raster)
        print(f"{os.path.basename(write_raster)} saved ({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())})")
    
    # 如果zonal_stat存在'zone_value'列，则删除该列
    if 'zone_value' in zonal_stat.columns:
        zonal_stat.drop(columns='zone_value', inplace=True)

    return zonal_stat