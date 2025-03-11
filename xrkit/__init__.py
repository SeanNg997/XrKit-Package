# import 
import xrkit.file_management
import xrkit.data_processing

# rename
get_files_in_directory = xrkit.file_management.get_files_in_directory
print_file_save_time = xrkit.file_management.print_file_save_time
csv_to_shp = xrkit.data_processing.csv_to_shp
zonal_statistics = xrkit.data_processing.zonal_statistics
reproject_to_wgs84 = xrkit.data_processing.reproject_to_wgs84

# function list
function_list = {
    'get_files_in_directory': '获取目录中的文件名称或路径',
    'print_file_save_time': '打印文件保存时间',
    'csv_to_shp': 'csv文件转shp文件',
    'zonal_statistics': '分区统计',
    'reproject_to_wgs84': '投影到WGS84坐标系（对齐全局格网）'
}

# show functions
def show_functions():
    for i, (key, value) in enumerate(function_list.items()):
        print(f'{i+1}. {key}：{value}')
