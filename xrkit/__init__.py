# file_management.py 文件管理相关函数
from .file_management import get_files_in_directory, print_file_save_time

# data_processing.py 数据处理相关函数
from .data_processing import csv_to_shp, zonal_statistics

# basic information
__all__ = ['get_files_in_directory', 'print_file_save_time', 'csv_to_shp', 'zonal_statistics']
__version__ = 'xrkit-version: 1.0'

# show functions
def show_functions():
    for i, (key, value) in enumerate(function_list.items()):
        print(f'{i+1}. {key}：{value}')

# function list
function_list = {
    'get_files_in_directory': '获取目录中的文件名称或路径',
    'print_file_save_time': '打印文件保存时间',
    'csv_to_shp': 'csv文件转shp文件',
    'zonal_statistics': '分区统计'
}