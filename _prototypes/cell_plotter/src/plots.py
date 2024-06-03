import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from library.study_space import SpatialSpikeTrain2D
from library.maps.map_utils import disk_mask



def flat_disk_mask(rate_map):
    masked_rate_map = disk_mask(rate_map)
    masked_rate_map.data[masked_rate_map.mask] = np.nan
    return  masked_rate_map.data

def plot_cell_waveform(cell, data_dir):

    fig = WaveformTemplateFig()

    for i in range(4):
        ch = cell.signal[:,i,:]
        idx = np.random.choice(len(ch), size=200)
        waves = ch[idx, :]
        avg_wave = np.mean(ch, axis=0)

        fig.waveform_channel_plot(waves, avg_wave, str(i+1), fig.ax[str(i+1)])

    animal = cell.animal_id
    animal_id = animal.split('tet')[0]
    tet = animal.split('tet')[-1]
    session = cell.ses_key
    unit = cell.cluster.cluster_label

    title = str(animal) + '_' + str(session) + '_unit_' + str(unit)

    fig.f.suptitle(title, ha='center', fontweight='bold', fontsize='large')

    """ save """
    # create a dsave and an fprefix
    # save_dir = data_dir + r'/output/' + str(animal) + r'/' + str(session) + r'/'
    save_dir = data_dir
    fprefix = r'tetrode_{}_session_{}_unit_{}'.format(tet,session.split('_')[-1],unit)

    ftemplate_short = "{}.{}"
    fshort = ftemplate_short.format(fprefix, r'png')
    fp = os.path.join(save_dir, fshort)
    fig.f.savefig(fp, dpi=360.)
    plt.close(fig.f)

def plot_cell_rate_map(cell, isCylinder, data_dir):

    fig = RatemapTemplateFig()

    pos_obj = cell.session_metadata.session_object.get_position_data()['position']

    sst = cell.session_metadata.session_object.make_class(SpatialSpikeTrain2D, {'cell': cell, 'position': pos_obj})
    rate_map_obj = sst.get_map('rate') 
    rate_map, _ = rate_map_obj.get_rate_map()

    if isCylinder:
        rate_map = flat_disk_mask(rate_map)

    fig.rate_map_plot(rate_map, fig.ax['1'])

    animal = cell.animal_id
    animal_id = animal.split('tet')[0]
    tet = animal.split('tet')[-1]
    session = cell.ses_key
    unit = cell.cluster.cluster_label

    title = str(animal) + '_' + str(session) + '_unit_' + str(unit)

    fig.f.suptitle(title, ha='center', fontweight='bold', fontsize='large')

    """ save """
    # create a dsave and an fprefix
    # save_dir = data_dir + r'/output/' + str(animal) + r'/' + str(session) + r'/'
    save_dir = data_dir
    fprefix = r'tetrode_{}_session_{}_unit_{}'.format(tet,session.split('_')[-1],unit)

    ftemplate_short = "{}.{}"
    fshort = ftemplate_short.format(fprefix, r'png')
    fp = os.path.join(save_dir, fshort)
    fig.f.savefig(fp, dpi=360.)
    plt.close(fig.f)


class WaveformTemplateFig():
    def __init__(self):
        self.f = plt.figure(figsize=(12, 6))
        # mpl.rc('font', **{'size': 20})


        self.gs = {
            'all': gridspec.GridSpec(1, 4, left=0.05, right=0.95, bottom=0.1, top=0.85, figure=self.f),
        }

        self.ax = {
            '1': self.f.add_subplot(self.gs['all'][:, :1]),
            '2': self.f.add_subplot(self.gs['all'][:, 1:2]),
            '3': self.f.add_subplot(self.gs['all'][:, 2:3]),
            '4': self.f.add_subplot(self.gs['all'][:, 3:]),
        }

    def waveform_channel_plot(self, waveforms, avg_waveform, channel, ax):

        ax.plot(waveforms.T, color='grey')

        ax.plot(avg_waveform, c='k', lw=2)

        ax.set_title('Channel ' + str(int(channel)))

class RatemapTemplateFig():
    def __init__(self):
        self.f = plt.figure(figsize=(6, 6))
        # mpl.rc('font', **{'size': 20})


        self.gs = {
            'all': gridspec.GridSpec(1, 1, left=0.05, right=0.95, bottom=0.1, top=0.85, figure=self.f),
        }

        self.ax = {
            '1': self.f.add_subplot(self.gs['all'][:, :]),
        }

    def rate_map_plot(self, rate_map, ax):

        img = ax.imshow(np.uint8(cm.jet(rate_map)*255))

        self.f.colorbar(img, ax=ax, fraction=0.046, pad=0.04)





