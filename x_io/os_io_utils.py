import os

def get_files_in_dir(dir_path):
    """
    Get all files in a directory.
    """
    return [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

def with_open_file(function, file_path, method='r'):
    """
    Open a file and run a function on it.
    """
    with open(file_path, method) as f:
        return function(f)