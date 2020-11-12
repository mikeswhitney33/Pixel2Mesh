import os


def get_test_dir():
    return os.path.dirname(os.path.realpath(__file__))


def get_src_dir():
    test_dir = get_test_dir()
    parent, _ = os.path.split(test_dir)
    return parent


def get_data_dir():
    src_dir = get_src_dir()
    return os.path.join(src_dir, "Data")


def get_external_dir():
    src_dir = get_src_dir()
    return os.path.join(src_dir, "external")
