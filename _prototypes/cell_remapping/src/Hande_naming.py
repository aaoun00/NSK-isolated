def split_filename(filename):
    # filename : M1_B6_4mo_session1_B_1m_lineartrack_042024_1050um

    name = filename.split('_')[0]
    date = filename.split('_')[7]
    assert len(date) == 6
    date = date[:4] + '20' + date[4:]
    assert len(date) == 8
    depth = filename.split('_')[8].split('um')[0]
    stim = None 

    return stim, depth, name, date


def extract_name_hande(filename):
    name = filename.split('_')[0]
    return name

Hande_naming_format = {
                    r'^([^-]+)\_([^-]+)\_([^-]+)\_([^-]+)\_([^-]+)\_([^-]+)\_([^-]+)\_([^-]+)\_([^-]+)$': split_filename,
}
