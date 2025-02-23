import os
import time

def get_files_in_directory(folder_path=os.getcwd(), extension='', rootpath=False):
    """
    20250113
    Get all files in the folder with the specified extension.
    
    :param folder_path: The folder path.
    :param end: The extension of the file.
    :param rootpath: Whether to include the root path.
    :return num_filesList: The number of files.
    :return filesList: The list of files.
    """

    # Initialize the list of files.
    filesList = []

    # If the folder does not exist, return the number of files and the list of files.
    if not os.path.exists(folder_path):
        print("The folder does not exist.")
        return filesList
    
    # If the extension is not empty, get the file list with the specified extension.
    if extension != '':

        # If the extension does not start with '.', add '.' to the extension.
        if extension[0] != '.':
            extension = '.' + extension
        
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(extension):  # 忽略大小写
                    filesList.append(file)

    # If the extension is empty, get all the files.
    else:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                filesList.append(file)

    # If rootpath is True, add the root path to the file name.
    if rootpath:
        filesList = [os.path.join(folder_path, file) for file in filesList]
        
    # Print the number of files.
    print("The number of files: ", len(filesList))
    
    # Return the number of files and the list of files.
    return filesList

def print_file_save_time(filename = 'File'):
    """
    20250114
    Print the time when the file is saved.
    
    :param filename: The name of the file.
    """
    print(f"{filename} saved ({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())})")