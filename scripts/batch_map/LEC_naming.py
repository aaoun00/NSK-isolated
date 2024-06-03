import pandas as pd


# def split_B6_LEC_20160404(filename):
#     name = filename.split('_')[0]
#     date = filename.split('_')[1].split('-')[0]
#     angle = filename.split('_')[1].split('-')[-1]

#     depth = _merge_depth(name, date)

#     return _check_angle(angle), depth, name, date

def _check_angle(angle):
    valid_angles = ['0','90','180','270','NO','noobject','no','zero', 'NO2', 'no2', '0_2','0_1','90_2','90_1','180_2',
                    '180_1','270_2','270_1', 'NO_1', 'NO_2','position0', 'position90', 'position180', 'position270', '0banana', 'NOobject']
    conv_angles = {'noobject': 'NO','no': 'NO','zero': '0', 'NO2': 'NO', 'no2': 'NO', 'NOobject': 'NO',
                   'position0': '0', 'position90': '90', 'position180': '180', 'position270': '270',
                   '0banana': '0', 
                   '0_2': '0', '0_1': '0', '90_2': '90', '90_1': '90', '180_2': '180', '180_1': '180', '270_2': '270', '270_1': '270', 'NO_1': 'NO', 'NO_2': 'NO'}

    assert angle in valid_angles, 'Invalid angle: {}'.format(angle)

    if angle in conv_angles.keys():
        angle = conv_angles[angle]

    return angle

def _check_odor(odor):
    valid_odors = ['xab','xba','abx','axb','bxa','bax']

    assert odor.lower() in valid_odors, 'Invalid odor order {}'.format(odor)

    return odor

def _merge_depth(name, date):
    # print(name)
    if name == 'B6-LEC1':
        # pth = r"C:\Users\aaoun\OneDrive - cumc.columbia.edu\Desktop\HussainiLab\neuroscikit_test_data\LEC_filter_excel\B6 LEC1 depths.xlsx"
        pth = r"/home/apollo/Documents/neuroscikit/B6 LEC1 depths.xlsx"
        df = pd.read_excel(pth)
        
        format_date = str(date[:4] + '/' + date[4:6] + '/' + date[6:])

    elif name == 'B6-LEC2': 
        pth = r"/home/apollo/Documents/neuroscikit/B6 LEC2 depths.xlsx"
        # pth = r"C:\Users\aaoun\OneDrive - cumc.columbia.edu\Desktop\HussainiLab\neuroscikit_test_data\LEC_filter_excel\B6 LEC2 depths.xlsx"
        df = pd.read_excel(pth)

        format_date = str(date[:4] + '/' + date[4:6] + '/' + date[6:])
    
    vals = df.loc[df['Session'] == format_date, 'Depth'].values
    datetime = pd.to_datetime(date)
    
    while len(vals) == 0:
        datetime -= pd.Timedelta(days=1)
        vals = df.loc[df['Session'] == datetime.strftime(r'%m/%d/%y'), 'Depth'].values
    depth = vals[0]

    return depth 

def split_name_date_int_int_angle(filename):
    # 'B6-LEC1_{date}-{int}-{int}-{angle}'
    angle = filename.split('-')[-1] 

    name = filename.split('_')[0]
    date = filename.split('_')[1].split('-')[0]

    depth = _merge_depth(name, date)

    return _check_angle(angle), depth, name, date

def split_name_date_angle_angle(filename):
    angle1 = filename.split('-')[-2]
    angle2 = filename.split('-')[-1]

    # angle = _check_angle(angle1) + '_' + 
    angle = _check_angle(angle2)

    name = filename.split('_')[0]
    date = filename.split('_')[1].split('-')[0]

    depth = _merge_depth(name, date)

    return angle, depth, name, date

def split_name_date_angle(filename):
    angle = filename.split('-')[-1]
    angle = _check_angle(angle)

    name = filename.split('_')[0]
    date = filename.split('_')[1].split('-')[0]

    depth = _merge_depth(name, date)

    return angle, depth, name, date

def split_name_date_position_angle(filename):
    angle = filename.split('position')[-1]

    name = filename.split('_')[0]
    date = filename.split('_')[1].split('-')[0]

    depth = _merge_depth(name, date)

    return _check_angle(angle), depth, name, date

def split_name_date_angle_angle_angle(filename):
    angle1 = filename.split('-')[-3]
    angle2 = filename.split('-')[-2]
    angle3 = filename.split('-')[-1]

    # angle = _check_angle(angle1) + '_' + _check_angle(angle2) + '_' + 
    angle = _check_angle(angle3)

    name = filename.split('_')[0]
    date = filename.split('_')[1].split('-')[0]

    depth = _merge_depth(name, date)

    return angle, depth, name, date

def split_name_date_int_angle_angle_angle(filename):
    angle1 = filename.split('-')[-3]
    angle2 = filename.split('-')[-2]
    angle3 = filename.split('-')[-1]

    # angle = _check_angle(angle1) + '_' + _check_angle(angle2) + '_' +
    angle = _check_angle(angle3)

    name = filename.split('_')[0]
    date = filename.split('_')[1].split('-')[0]

    depth = _merge_depth(name, date)

    return angle, depth, name, date

def split_name_date_angle_word(filename):
    name = filename.split('_')[0]
    date = filename.split('_')[1].split('-')[0]

    if 'banana' in filename:
        angle = filename.split('banana')[0].split('-')[-1]
    elif 'APLLE' in filename:
        angle = filename.split('APLLE')[0].split('-')[-1]
    if 'set' in filename:
        angle = filename.split('set')[0].split('-')[-1]

    depth = _merge_depth(name, date)

    return _check_angle(angle), depth, name, date

def split_name_date_angle_apple_angle_angle(filename):
    angle2 = filename.split('APPLEmovedit')[-1].split('-')[0]
    angle3 = filename.split('APPLEmovedit')[-1].split('-')[1]
    angle1 = filename.split('APPLEmovedit')[0].split('-')[-1]

    # angle = _check_angle(angle1) + '_' + _check_angle(angle2) + '_' + 
    angle = _check_angle(angle3)

    name = filename.split('_')[0]
    date = filename.split('_')[1].split('-')[0]

    depth = _merge_depth(name, date)

    return angle, depth, name, date

def split_name_date_angle_angle_int(filename):
    angle1 = filename.split('_')[0].split('-')[-1]
    angle2 = filename.split('_')[0].split('-')[-2]

    # angle = _check_angle(angle1) + '_' + 
    angle = _check_angle(angle2) 

    name = filename.split('_')[0]
    date = filename.split('_')[1].split('-')[0]

    depth = _merge_depth(name, date)

    return angle, depth, name, date

def split_name_date_angle_depth(filename):
    angle = filename.split('-')[-2]
    depth = filename.split('-')[-1]
    name = filename.split('_')[0]
    date = filename.split('_')[1].split('-')[0]

    return _check_angle(angle), depth, name, date

# def split_name_date_angleint_depth(filename):
#     angle = str(filename.split('-')[-2])[:2]
#     depth = filename.split('-')[-1]
#     name = filename.split('_')[0]
#     date = filename.split('_')[1].split('-')[0]

#     return _check_angle(angle), depth, name, date


def split_name_date_depth_word_angle(filename):
    if 'object' in filename and 'noobject' not in filename:
        angle = filename.split('object')[-1]
    depth = filename.split('_')[1].split('-')[-2]
    name = filename.split('_')[0]
    date = filename.split('_')[1].split('-')[0]
    
    return _check_angle(angle), depth, name, date

def split_name_date_depth_noodor_noobject(filename):
    angle = 'NO'
    depth = filename.split('-')[-3]
    name = filename.split('_')[0]
    date = filename.split('_')[1].split('-')[0]

    return _check_angle(angle), depth, name, date

def split_name_date_depth_odor_order(filename):
    # 'ANT-119a-6_{date}-{depth}-odor{order}'
    odor = filename.split('odor')[-1]
    depth = filename.split('-')[-2]
    date = filename.split('_')[1].split('-')[0]
    name = filename.split('_')[0]

    return _check_odor(odor), depth, name, date

def split_name_date_round_depth_angle(filename):
    # 'B6-LEC1_{date}-ROUND-{depth}-{angle}'
    angle = filename.split('-')[-1]
    depth = filename.split('-')[-2]
    name = filename.split('_')[0]
    date = filename.split('_')[1].split('-')[0]

    return _check_angle(angle), depth, name, date

def split_name_name_date_round_depth_angle(filename):
    # 'NON-73-6_NON-73-6_{date}-ROUND-{depth}-{angle}'
    angle = filename.split('-')[-1]
    depth = filename.split('-')[-2]
    name = filename.split('_')[0] 
    date = filename.split('_')[-1].split('-')[0]

    return _check_angle(angle), depth, name, date

def split_name_name_date_round_depth_angle_int(filename):
    # 'NON-73-6_NON-73-6_{date}-ROUND-{depth}-{angle}-{int}'
    angle = filename.split('-')[-2]
    depth = filename.split('-')[-3]
    name = filename.split('_')[0] 
    date = filename.split('_')[-1].split('-')[0]

    return _check_angle(angle), depth, name, date

def split_name_date_round_depth_angle_int(filename):
    # 'ANT-133a-4_{date}-ROUND-{depth}-{angle}-{int}'
    angle = filename.split('-')[-2]
    depth = filename.split('-')[-3]
    date = filename.split('_')[1].split('-')[0]
    name = filename.split('_')[0]

    return _check_angle(angle), depth, name, date

def split_name_date_round_depth_odor_int(filename):
    # 'ANT-133a-4_{date}-ROUND-{depth}-1{o}2{o}3{o}-{int}'
    o1 = filename.split('-')[-2].split('1')[-1].split('2')[0]
    o2 = filename.split('-')[-2].split('2')[-1].split('3')[0]
    o3 = filename.split('-')[-2].split('3')[-1]

    odor = o1 + o2 + o3

    depth = filename.split('-')[-3]
    name = filename.split('_')[0]
    date = filename.split('_')[1].split('-')[0]

    return _check_odor(odor), depth, name, date

def split_name_date_round_depth_odor(filename):
    # 'ANT-133a-4_{date}-ROUND-{depth}-1{o}2{o}3{o}'
    o1 = filename.split('-')[-1].split('1')[-1].split('2')[0]
    o2 = filename.split('-')[-1].split('2')[-1].split('3')[0]
    o3 = filename.split('-')[-1].split('3')[-1]

    odor = o1 + o2 + o3

    depth = filename.split('-')[-2]
    name = filename.split('_')[0]
    date = filename.split('_')[1].split('-')[0]

    return _check_odor(odor), depth, name, date

def split_name_date_depth_word_angle_word(filename):
    # 'ANT-120-4_{date}-{depth}-object{angle}-foundmoved',

    if 'object' in filename and 'noobject' not in filename:
        angle = filename.split('object')[-1].split('-')[0]
    depth = filename.split('_')[1].split('-')[-3]
    name = filename.split('_')[0]
    date = filename.split('_')[1].split('-')[0]

    return _check_angle(angle), depth, name, date

def split_name_date_round_depth_angle(filename):
    # 'ANT-133a-4_{date}-ROUND-{depth}-{angle}'
    angle = filename.split('-')[-1]
    depth = filename.split('-')[-2]
    name = filename.split('_')[0]
    date = filename.split('_')[1].split('-')[0]

    return _check_angle(angle), depth, name, date

def split_name_date_round_depth_angle_angle(filename):
    # 'ANT-133a-4_{date}-ROUND-{depth}-{angle}-{angle}'
    angle1 = filename.split('-')[-2]
    angle2 = filename.split('-')[-1]
    # angle = _check_angle(angle1) + '_' + _check_angle(angle2) 
    angle = _check_angle(angle2)
    depth = filename.split('-')[-3]
    name = filename.split('_')[0]
    date = filename.split('_')[1].split('-')[0]

    return angle, depth, name, date

def split_name_date_round_depth_angle_int(filename):
    # 'ANT-133a-4_{date}-ROUND-{depth}-{angle}-{int}'
    angle = filename.split('-')[-2]
    depth = filename.split('-')[-3]
    name = filename.split('_')[0]
    date = filename.split('_')[1].split('-')[0]

    return _check_angle(angle), depth, name, date

def split_name_date_round_depth_angle_int_word(filename):
    # 'ANT-133a-4_{date}-ROUND-{depth}-{angle}-{int}-drilling'
    angle = filename.split('-')[-3]
    depth = filename.split('-')[-4]
    name = filename.split('_')[0]
    date = filename.split('_')[1].split('-')[0]

    return _check_angle(angle), depth, name, date

def split_name_date_angle_depth_int(filename):
    # 'B6-LEC1_{date}-{angle}-{depth}-{int}'
    angle = filename.split('-')[-3]
    depth = filename.split('-')[-2]
    date = filename.split('_')[1].split('-')[0]
    name = filename.split('_')[0]

    return _check_angle(angle), depth, name, date

def split_name_date_word_angle_depth(filename):
    # 'B6-1M_{date}_ROUND-{angle}-{depth}'
    depth = filename.split('-')[-1]
    date = filename.split('_')[1]
    name = filename.split('_')[0]
    angle = filename.split('-')[-2]

    return _check_angle(angle), depth, name, date

def split_name_date_word_angle(filename):
    # B6-LEC1_20151211-cylinder-0
    angle = filename.split('-')[-1]
    date = filename.split('_')[1].split('-')[0]
    name = filename.split('_')[0]

    depth = _merge_depth(name, date)

    return _check_angle(angle), depth, name, date

def split_name_date_depth_angle(filename):
    # '{animal}_{date}_{depth}-{angle}'
    angle = filename.split('-')[-1]
    depth = filename.split('-')[-2]
    date = filename.split('_')[1]
    name = filename.split('_')[0]

    return _check_angle(angle), depth, name, date

def extract_name_lec(filename):
    name = filename.split('_')[0]
    # print(filename, name)
    if 'B6' in name:
        group = 'B6'
    elif 'ANT' in name:
        group = 'ANT'
    elif 'NON' in name:
        group = 'NON'
    return group, name

LEC_naming_format = {
    'B6': {
            'B6-LEC1': {
                'object': {
                    r'^B6-LEC1_[0-9]{8}\-([0-9]+)([a-zA-Z]+)([^-]+)\-([^-]+)$': split_name_date_angle_apple_angle_angle, # 'B6-LEC1_{date}-{angle}APPLEmovedit{angle}-{angle}',
                    r'^B6-LEC1_[0-9]{8}\-cylinder\-([^-]+)$': split_name_date_word_angle, # B6-LEC1_20151211-cylinder-0
                    r'^B6-LEC1_[0-9]{8}\-[0-9]{1}\-[0-9]{1}\-([^-]+)$': split_name_date_int_int_angle, # 'B6-LEC1_{date}-{int}-{int}-{angle}': 
                    r'^B6-LEC1_[0-9]{8}\-([^-]+)\-([^-]+)$':  split_name_date_angle_angle, # split_name_date_angle_angle 'B6-LEC1_{date}-{angle}-{angle}
                    r'^B6-LEC1_[0-9]{8}\-([^-]+)$': split_name_date_angle, # 'B6-LEC1_{date}-{angle}',
                    r'^B6-LEC1_[0-9]{8}\-position([^-]+)$': split_name_date_position_angle, # 'B6-LEC1_{date}-position{angle}',
                    r'^B6-LEC1_[0-9]{8}\-([^-]+)\-([^-]+)\-([^-]+)$': split_name_date_angle_angle_angle, # 'B6-LEC1_{date}-{angle}-{angle}-{angle}',
                    r'^B6-LEC1_[0-9]{8}\-([0-9]+)banana$': split_name_date_angle_word, # 'B6-LEC1_{date}-{angle}banana',
                    r'^B6-LEC1_[0-9]{8}\-[0-9]{1}\-([^-]+)\-([^-]+)\-([^-]+)$': split_name_date_int_angle_angle_angle, # 'B6-LEC1_{date}-{int}-{angle}-{angle}-{angle}',
                },
                'odor': {
    
                },
            },  
            'B6-LEC2': {
                'object': {
                    r'^B6-LEC2_[0-9]{8}\-[0-9]{1}\-[0-9]{1}\-([^-]+)$': split_name_date_int_int_angle, # 'B6-LEC2_{date}-{int}-{int}-{angle}',
                    r'^B6-LEC2_[0-9]{8}\-([^-]+)\-([^-]+)$':  split_name_date_angle_angle, # 'B6-LEC2_{date}-{angle}-{angle}',
                    r'^B6-LEC2_[0-9]{8}\-([^-]+)$': split_name_date_angle, # 'B6-LEC2_{date}-{angle}',
                    # r'^B6-LEC2_[0-9]{8}\-([^-]+)\-([^-]+)\_[0-9]+$': split_name_date_angle_angle_int, # 'B6-LEC2_{date}-{angle}-{angle}_{int}',
                    r'^B6-LEC2_[0-9]{8}\-([0-9]+)set$': split_name_date_angle_word, # 'B6-LEC2_{date}-{angle}set',
                    r'^B6-LEC2_[0-9]{8}\-([0-9]+)banana$': split_name_date_angle_word, # 'B6-LEC2_{date}-{angle}banana',
                    r'^B6-LEC2_[0-9]{8}\-([0-9]+)APPLE$': split_name_date_angle_word, # 'B6-LEC2_{date}-{angle}APPLE',
                    r'^B6-LEC2_[0-9]{8}\-([^-]+)\-([^-]+)\-([^-]+)$': split_name_date_angle_angle_angle, # 'B6-LEC2_{date}-{angle}-{angle}-{angle}',
                },
                'odor': {
    
                },
            },
            'B6-1M': {
                'object':{
                    r'^B6-1M_[0-9]{8}\-([^-]+)\-([^-]+)$': split_name_date_angle_depth, # 'B6-1M_{date}-{angle}-{depth}',
                    r'^B6-1M_[0-9]{8}\-([^-]+)\-([0-9]+)\-([0-9]+)$': split_name_date_angle_depth_int, # 'B6-1M_{date}-{angle}-{depth}-{int}',
                    r'^B6-1M_(\d{8})_([a-zA-Z]+-[^-]+-[0-9]+)$': split_name_date_word_angle_depth, # 'B6-1M_{date}-{angle}-{depth}-{int}',
                },
                'odor': {
    
                },
            },        
            'B6-2M': {
                'object':{
                    # r'^B6-2M_[0-9]{8}\-[a-zA-Z]{2}[0-9]{1}\-([^-]+)$': split_name_date_angleint_depth, # 'B6-2M_{date}-{angle}{int}-{depth}',
                    r'^B6-2M_[0-9]{8}\-([^-]+)\-([^-]+)$': split_name_date_angle_depth, # 'B6-2M_{date}-{angle}-{depth}',
                },
                'odor': {

                },
            },
    },
    'ANT': {
        'ANT-119a-6':{
            'object': {
                r'^ANT-119a-6_[0-9]{8}\-([^-]+)\-object([^-]+)$': split_name_date_depth_word_angle, # 'ANT-119a-6_{date}-{depth}-object{angle}',
                r'^ANT-119a-6_[0-9]{8}\-([^-]+)\-noodor-noobject$': split_name_date_depth_noodor_noobject, # 'ANT-119a-6_{date}-{depth}-noodor-noobject',
                r'^ANT-119a-6_[0-9]{8}\-([^-]+)\-([^-]+)$': split_name_date_depth_angle, # 'ANT-119a-6_{date}-{depth}-{angle}',
            },
            'odor': {
                r'^ANT-119a-6_[0-9]{8}\-([^-]+)\-odor([^-]+)$': split_name_date_depth_odor_order, # 'ANT-119a-6_{date}-{depth}-odor{order}',
    
            },
        },
        'ANT-133a-4':{
            'object': {
                r'^ANT-133a-4_[0-9]{8}\-ROUND\-([^-]+)\-([^-]+)$': split_name_date_round_depth_angle, # 'ANT-133a-4_{date}-ROUND-{depth}-{angle}',
                r'^ANT-133a-4_[0-9]{8}\-ROUND\-([^-]+)\-([^-]+)\-([0-9]+)$': split_name_date_round_depth_angle_int, # 'ANT-133a-4_{date}-ROUND-{depth}-{angle}-{int}',
            },
            'odor': {
                r'^ANT-133a-4_[0-9]{8}\-ROUND\-([^-]+)\-1[a-zA-Z]{1}2[a-zA-Z]{1}3[a-zA-Z]{1}\-[0-9]{1}$': split_name_date_round_depth_odor_int, # 'ANT-133a-4_{date}-ROUND-{depth}-1{o}2{o}3{o}-{int}',
                r'^ANT-133a-4_[0-9]{8}\-ROUND\-([^-]+)\-1[a-zA-Z]{1}2[a-zA-Z]{1}3[a-zA-Z]{1}$': split_name_date_round_depth_odor, # 'ANT-133a-4_{date}-ROUND-{depth}-1{o}2{o}3{o}',
                r'^ANT-133a-4_[0-9]{8}\-ROUND\-([^-]+)\-N1[a-zA-Z]{1}2[a-zA-Z]{1}3[a-zA-Z]{1}$': split_name_date_round_depth_odor # 'ANT-133a-4_{date}-ROUND-{depth}-N1{o}2{o}3{o}',
            },
        },
        'ANT-135a-7':{
            'object': {
                r'^ANT-135a-7_[0-9]{8}\-ROUND\-([^-]+)\-([^-]+)$': split_name_date_round_depth_angle, # 'ANT-135a-7_{date}-ROUND-{depth}-{angle}',
                r'^ANT-135a-7_[0-9]{8}\-ROUND\-([^-]+)\-([^-]+)\-([0-9]+)$': split_name_date_round_depth_angle_int, # 'ANT-135a-7_{date}-ROUND-{depth}-{angle}-{int}',
            },
            'odor': {
                r'^ANT-135a-7_[0-9]{8}\-ROUND\-([^-]+)\-1[a-zA-Z]{1}2[a-zA-Z]{1}3[a-zA-Z]{1}-[0-9]{1}$': split_name_date_round_depth_odor_int, # 'ANT-135a-7_{date}-ROUND-{depth}-1{o}2{o}3{o}-{int}',
                r'^ANT-135a-7_[0-9]{8}\-ROUND\-([^-]+)\-1[a-zA-Z]{1}2[a-zA-Z]{1}3[a-zA-Z]{1}$': split_name_date_round_depth_odor, # 'ANT-135a-7_{date}-ROUND-{depth}-1{o}2{o}3{o}',
            },
        },
        'ANT-120-4':{
            'object': {
                r'^ANT-120-4_[0-9]{8}\-([^-]+)\-object([^-]+)$': split_name_date_depth_word_angle, # 'ANT-120-4_{date}-{depth}-object{angle}',
                r'^ANT-120-4_[0-9]{8}\-([^-]+)\-noodor-noobject$': split_name_date_depth_noodor_noobject, # 'ANT-120-4_{date}-{depth}--noodor-noobject',
                r'^ANT-120-4_[0-9]{8}\-([^-]+)\-object([^-]+)\-foundmoved$': split_name_date_depth_word_angle_word, # 'ANT-120-4_{date}-{depth}-object{angle}-foundmoved',
                r'^ANT-120-4_[0-9]{8}\-ROUND\-([^-]+)\-([^-]+)\-([0-9]+)$': split_name_date_round_depth_angle_int, # 'ANT-135a-7_{date}-ROUND-{depth}-{angle}-{int}',
           },
            'odor': {
                r'^ANT-120-4_[0-9]{8}\-([^-]+)\-odor([^-]+)$': split_name_date_depth_odor_order, # 'ANT-120-4_{date}-{depth}-odor{order}',
            },
        },
        'ANT-140-4':{
            'object': {
                r'^ANT-140-4_[0-9]{8}\-ROUND\-([^-]+)\-([^-]+)$': split_name_date_round_depth_angle, # 'ANT-140-4_{date}-ROUND-{depth}-{angle}',
                r'^ANT-140-4_[0-9]{8}\-ROUND\-([^-]+)\-([^-]+)\-[0-9]{1}$': split_name_date_round_depth_angle_int, # 'ANT-140-4_{date}-ROUND-{depth}-{angle}-{int}',
                r'^ANT-140-4_[0-9]{8}\-ROUND\-([^-]+)\-([^-]+)\-([^-]+)$': split_name_date_round_depth_angle_angle, # 'ANT-140-4_{date}-ROUND-{depth}-{angle}-{angle}',
            },
            'odor': {
                r'^ANT-140-4_[0-9]{8}\-ROUND\-([^-]+)\-1[a-zA-Z]{1}2[a-zA-Z]{1}3[a-zA-Z]{1}$': split_name_date_round_depth_odor, # 'ANT-140-4_{date}-ROUND-{depth}-1{o}2{o}3{o}',
            },
        },
    },
    'NON': {
        'NON-73-6': {
            'object': {
                r'^NON-73-6_[0-9]{8}\-ROUND\-([^-]+)\-([^-]+)$': split_name_date_round_depth_angle, # 'NON-73-6_{date}-ROUND-{depth}-{angle}',
                r'^NON-73-6_[0-9]{8}\-ROUND\-([^-]+)\-([^-]+)\-([0-9]+)$': split_name_date_round_depth_angle_int, # 'NON-73-6_{date}-ROUND-{depth}-{angle}-{int}',
                r'^NON-73-6_[0-9]{8}\-ROUND\-([^-]+)\-([^-]+)\-[0-9]{1}\-drilling$': split_name_date_round_depth_angle_int_word, # 'NON-73-6_{date}-ROUND-{depth}-{angle}-{int}-drilling',
                r'^NON-73-6_NON-73-6_[0-9]{8}\-ROUND\-([^-]+)\-([^-]+)$': split_name_name_date_round_depth_angle, # 'NON-73-6_NON-73-6_{date}-ROUND-{depth}-{angle}',
                r'^NON-73-6_NON-73-6_[0-9]{8}\-ROUND\-([^-]+)\-([^-]+)\-[0-9]{1}$': split_name_name_date_round_depth_angle_int, # 'NON-73-6_NON-73-6_{date}-ROUND-{depth}-{angle}',

            },
            'odor': {
                r'^NON-73-6_[0-9]{8}\-ROUND\-([^-]+)\-1[a-zA-Z]{1}2[a-zA-Z]{1}3[a-zA-Z]{1}$': split_name_date_round_depth_odor, # 'NON-73-6_{date}-ROUND-{depth}-1{o}2{o}3{o}',
            },
        },
        'NON-INT-01': {
            'object': {
                r'^NON-INT-01_[0-9]{8}\-([^-]+)\-ROUND\-([^-]+)\-([0-9]+)$': split_name_date_angle_depth_int, # 'NON-INT-01_{date}-{angle}-{depth}-{int}',
                r'^NON-INT-01_[0-9]{8}\-ROUND\-([^-]+)\-([^-]+)$': split_name_date_angle_depth, # 'NON-INT-01_{date}-{angle}-{depth}'
                r'^NON-INT-01_[0-9]{8}\-([^-]+)\-([^-]+)\-([0-9]+)$': split_name_date_angle_depth_int,
                r'^NON-INT-01_[0-9]{8}\-([^-]+)\-([^-]+)$': split_name_date_angle_depth,
            },
            'odor': {
    
            },
        },
        'NON-INT-02': {
            'object': {
                r'^NON-INT-02_[0-9]{8}\-([^-]+)\-ROUND\-([^-]+)\-([0-9]+)$': split_name_date_angle_depth_int, # 'NON-INT-01_{date}-{angle}-{depth}-{int}',
                r'^NON-INT-02_[0-9]{8}\-ROUND\-([^-]+)\-([^-]+)$': split_name_date_angle_depth, # 'NON-INT-02_{date}-{angle}-{depth}',
                r'^NON-INT-02_[0-9]{8}\-([^-]+)\-([^-]+)$': split_name_date_angle_depth,
            },
            'odor': {
    
            },
        },
        'NON-INT-03': {
            'object': {
                r'^NON-INT-03_[0-9]{8}\-([^-]+)\-ROUND\-([^-]+)\-([0-9]+)$': split_name_date_angle_depth_int, # 'NON-INT-03_{date}-{angle}-{depth}-{int}',
                r'^NON-INT-03_[0-9]{8}\-ROUND\-([^-]+)\-([^-]+)$': split_name_date_angle_depth,# 'NON-INT-03_{date}-{angle}-{depth}'
                r'^NON-INT-03_[0-9]{8}\-([^-]+)\-([^-]+)$': split_name_date_angle_depth,
                r'^NON-INT-03_[0-9]{8}\-([^-]+)\-([^-]+)\-([0-9]+)$': split_name_date_angle_depth_int,
            },
            'odor': {
    
            },
        },
        'NON-88-1': {
            'object': {
                r'^NON-88-1_[0-9]{8}\-ROUND\-([^-]+)\-([^-]+)$': split_name_date_round_depth_angle, # 'NON-88-1_{date}-ROUND-{depth}-{angle}',
                r'^NON-88-1_[0-9]{8}\-ROUND\-([^-]+)\-([^-]+)\-([0-9]+)$': split_name_date_round_depth_angle_int, # 'NON-88-1_{date}-ROUND-{depth}-{angle}-{int}',
            },
            'odor': {
                r'^NON-88-1_[0-9]{8}\-ROUND\-([^-]+)\-1[a-zA-Z]{1}2[a-zA-Z]{1}3[a-zA-Z]{1}$': split_name_date_round_depth_odor, # 'NON-88-1_{date}-ROUND-{depth}-1{o}2{o}3{o}',
            },
        }
    },
}

