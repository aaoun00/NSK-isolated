def split_filename(filename):
    # filename : L-10262021-mAPPKI-PHF-1uL-HPC-950um-900s-Rectangle-No-Object-Evening-TINT

    date = filename.split('-')[1]
    name = filename.split('-')[0]
    # index of the item with um in it
    split_item_index = [i for i, item in enumerate(filename.split('-')) if 'um' in str(item.lower())]
    assert len(split_item_index) == 1, f'Found more than one item or no items with um in it: {split_item_index}'
    depth = filename.split('-')[split_item_index[0]]
    blank = None

    return blank, depth, name, date


def extract_name_lc(filename):
    name = filename.split('_')[0]
    return name

LC_naming_format = {
                    # filename: L-10262021-mAPPKI-PHF-1uL-HPC-950um-900s-Rectangle-No-Object-Evening-TINT
                    # SPLIT BY DASHES
                    r'^([^-]+)\-([0-9]{8})\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)$': split_filename,
                    r'^([^-]+)\-([0-9]{8})\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)$': split_filename,
                    r'^([^-]+)\-([0-9]{8})\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)$': split_filename,
                    r'^([^-]+)\-([0-9]{8})\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)$': split_filename,
                    r'^([^-]+)\-([0-9]{8})\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)$': split_filename,
                    r'^([^-]+)\-([0-9]{8})\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)$': split_filename,
                    r'^([^-]+)\-([0-9]{8})\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)$': split_filename,
                    r'^([^-]+)\-([0-9]{8})\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)$': split_filename,
                    r'^([^-]+)\-([0-9]{8})\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)$': split_filename,
                    r'^([^-]+)\-([0-9]{8})\-([^-]+)\-([^-]+)\-([^-]+)\-([^-]+)$': split_filename,
                    r'^([^-]+)\-([0-9]{8})\-([^-]+)\-([^-]+)\-([^-]+)$': split_filename,
                    # r'^([^-]+)\_[0-9]{8}\-[0-9]{2}\-([^-]+)\-([^-]+)\-([^-]+)$': split_filename,
}
