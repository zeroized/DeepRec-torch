def get_file_dir_and_name(path):
    last_sep = path.rfind('/')
    dir = path[:last_sep]
    name = path[last_sep + 1:]
    return dir, name
