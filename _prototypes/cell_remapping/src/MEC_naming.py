import re

def split_filename(filename):
    print(filename)
    name = filename.split('_')[0]
    date = filename.split('_')[1].split('-')[0]
    depth = filename.split('-')[-2]
    blank = None

    print(blank, depth, name, date)

    return blank, depth, name, date


def extract_name_mec(filename):
    name = filename.split('_')[0]
    return name

MEC_naming_format = {
                    # r'^([^-]+)\_[0-9]{8}\-[0-9]{2}\-([^-]+)\-([^-]+)\-([^-]+)$': split_filename,
                    r'^([^-]+)-([^-]+)_([0-9]{8})-[0-9]{2}-([^-]+)-([^-]+)-([^-]+)$': split_filename,
                    r'^([^-]+)_([0-9]{8})-[0-9]{2}-([^-]+)-([^-]+)-([^-]+)$': split_filename,
                    r'^([^-]+)-([^-]+)_([0-9]{8})-[0-9]{1}-([^-]+)-([^-]+)-([^-]+)$': split_filename,
                    r'^([^-]+)_([0-9]{8})-[0-9]{1}-([^-]+)-([^-]+)-([^-]+)$': split_filename,
}
