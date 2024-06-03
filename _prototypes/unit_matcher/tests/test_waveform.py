import os
import sys
PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

import pytest
import _prototypes.unit_matcher.tests.read as read

from _prototypes.unit_matcher.waveform import (
    time_index
    ,derivative
    ,derivative2
    ,zero_crossings
    ,local_extrema
    ,area_under_curve
    ,symmetric_logarithm
    ,Point
    ,morphological_points
    ,waveform_features
)

from matplotlib.pyplot import plot, legend, show, axvline, axhline
import matplotlib.pyplot as plt
import numpy as np

waveform = read.waveform
time_step = read.time_step

# Domain Conversion Functions

def test_time_index():
    assert len(waveform) == len(time_index(waveform, time_step))
    #plot(time_index(waveform, time_step), waveform)

def test_derivative():
    assert len(waveform) == len(derivative(waveform, time_step))
    #plot(time_index(waveform, time_step), derivative(waveform, time_step), linestyle='--')

def test_derivative2():
    assert len(waveform) == len(derivative2(waveform, time_step))
    #plot(time_index(waveform, time_step), derivative2(waveform, time_step), linestyle=':')


# Feature Utility Functions

@pytest.mark.skip(reason="Tested implicitly through morphological_points")
def test_local_extrema():
    d_waveform = derivative(waveform, time_step)
    extrema = local_extrema(waveform, time_step)
    d_extrema = local_extrema(d_waveform, time_step)
    #for e in extrema:
    #    plot(time_index(waveform, time_step)[e], d_waveform[e], 'o')
    #for e in d_extrema:
    #    axvline(time_index(waveform, time_step)[e], linestyle=':')
    #show()

# Key morphological point objects

def test_point():
    t = time_index(waveform, time_step)
    d_waveform = derivative(waveform, time_step)
    d2_waveform = derivative2(waveform, time_step)
    p = Point(10, t, waveform, d_waveform, d2_waveform)
    assert p.t == t[10]
    assert p.v == waveform[10]
    assert p.dv == d_waveform[10]
    assert p.d2v == d2_waveform[10]
    assert type(p) == Point
    assert type(p.i) == int
    assert type(p.t) == float
    assert type(p.v) == float
    assert type(p.dv) == float
    assert type(p.d2v) == float


def test_morphological_points():
    plot_waveform_points(waveform, time_step)

# Main Feature Extraction Function

def test_waveform_features():
    i=-1
    count = 0
    for wf in read.session_dict1['ch4']:
        i+=1
        try:
            feature_vector = waveform_features(wf, time_step)
        except:
            count += 1
    print(count, i, count/i)



def plot_waveform_points(waveform, time_step):
    t = time_index(waveform, time_step)
    d_waveform = derivative(waveform, time_step)
    d2_waveform = derivative2(waveform, time_step)

    p1, p2, p3, p4, p5, p6 = morphological_points(t, waveform, d_waveform, d2_waveform, time_step)

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('time (mS)')
    ax1.set_ylabel('mV', color=color)
    ax1.plot(t, waveform, color=color)
    ax1.plot(p1.t, p1.v, 'o', color=color, label='p1')
    ax1.plot(p3.t, p3.v, 'o', color=color, label='p3')
    ax1.plot(p5.t, p5.v, 'o', color=color, label='p5')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:orange'
    ax2.set_ylabel('mV/mS', color=color)  # we already handled the x-label with ax1
    ax2.plot(t, d_waveform, color=color, linestyle='--')
    ax2.plot(p2.t, p2.dv, 'o', color=color, label='p2')
    ax2.plot(p4.t, p4.dv, 'o', color=color, label='p4')
    ax2.plot(p6.t, p6.dv, 'o', color=color, label='p6')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.legend()
    plt.show()