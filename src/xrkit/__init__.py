# import 
import xrkit.file_management
import xrkit.data_processing

# rename
get_files_in_directory = xrkit.file_management.get_files_in_directory
print_file_save_time = xrkit.file_management.print_file_save_time
csv_to_shp = xrkit.data_processing.csv_to_shp
zonal_statistics = xrkit.data_processing.zonal_statistics
zonal_statistics_i = xrkit.data_processing.zonal_statistics_i
reproject_to_wgs84 = xrkit.data_processing.reproject_to_wgs84
reproject_to_wgs84_i = xrkit.data_processing.reproject_to_wgs84_i
aggregate = xrkit.data_processing.aggregate
aggregate_i = xrkit.data_processing.aggregate_i
parafun = xrkit.data_processing.parafun

# function list
function_list = {
    'get_files_in_directory': '获取目录中的文件名称或路径',
    'print_file_save_time': '打印文件保存时间',
    'parafun': '并行计算',
    'csv_to_shp': 'csv文件转shp文件',
    'zonal_statistics': '分区统计',
    'zonal_statistics_i': '分区统计（交互式）',
    'reproject_to_wgs84': '重投影到WGS84坐标系',
    'reproject_to_wgs84_i': '重投影到WGS84坐标系（交互式）',
    'aggregate': '聚合',
    'aggregate_i': '聚合（交互式）',
}

# show functions
def show_functions():
    for i, (key, value) in enumerate(function_list.items()):
        print(f'{i+1}. {key}：{value}')
