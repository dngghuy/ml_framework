import os


def create_fold_mapping_dict(num_folds):
    """
    Create a fold mapping dictionary
    """
    fold_mapping_dict = {key: [val for val in range(num_folds) if val != key] for key in range(5)}

    return fold_mapping_dict


def check_make_dirs(path_dir):
    """
    Check if path_dir exists; if not then create one
    """
    if not os.path.isdir(path_dir):
        os.makedirs(path_dir)
