import os
import time

def get_files_in_directory(folder_path=os.getcwd(),
                           extension='', 
                           rootpath=False):
    """
    Get a list of file names in a specified folder.
    
    Parameters
    ----------
    folder_path : str, optional
        The path of the folder (default is the current working directory).
    extension : str, optional
        The file extension to filter by (default is '' for no filtering).
    rootpath : bool, optional
        Whether to include the full path of files (default is False).
    
    Returns
    -------
    list
        A list of file names (with or without the full path, based on `rootpath`).
    
    Updated
    -------
    20250320
    """

    # If the folder does not exist, raise an error.
    if not os.path.isdir(folder_path):
        raise ValueError("The folder does not exist.")
    
    # initialization
    filesList = []

    if extension == '':
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                filesList.append(file)
    else:
        if extension[0] != '.':
            extension = '.' + extension
        
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(extension.lower()):
                    filesList.append(file)

    # If rootpath is True, add the root path to the file name.
    if rootpath:
        filesList = [os.path.join(folder_path, file) for file in filesList]
        
    # Print the number of files.
    print("The number of files: ", len(filesList))
    
    # Return the number of files and the list of files.
    return filesList

def print_file_save_time(filename: str):
    """
    Print the save time of the file.
    
    Parameters
    ----------
    filename : str
        The name of the file.
    
    Returns
    -------
    none
    
    Updated
    -------
    20250320
    """
    
    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f"{filename} saved ({t})")