"""Adapted from autoencoded-vocal-analysis/examples/mouse_sylls_mwe.py"""
from itertools import repeat

from joblib import Parallel, delayed
import numpy as np
import os

from ava.data.data_container import DataContainer
from ava.models.vae import X_SHAPE, VAE
from ava.segmenting.segment import segment
from ava.segmenting.amplitude_segmentation import get_onsets_offsets


params = {
	'segment': {
		'min_freq': 30e3, # minimum frequency
		'max_freq': 110e3, # maximum frequency
		'nperseg': 1024, # FFT
		'noverlap': 512, # FFT
		'spec_min_val': -10.0, # minimum log-spectrogram value
		'spec_max_val': 2.0, # maximum log-spectrogram value
		'fs': 250000, # audio samplerate
		'th_1':0.295, # segmenting threshold 1
		'th_2':0.3, # segmenting threshold 2
		'th_3':0.305, # segmenting threshold 2
		'max_dur': 0.2, # maximum syllable duration
		'min_dur':0.03, # minimum syllable duration
		'smoothing_timescale': 0.007, # timescale for smoothing amplitude trace
		'softmax': True, # puts amplitude values in [0,1]
		'temperature': 0.5, # temperature parameter for softmax
		'algorithm': get_onsets_offsets, # segmentation algorithm
	},
}

root = '/home/pimienta/Documents/data/vocal/goffinet'
audio_dirs = [os.path.join(root, 'BM003')]
seg_dirs = [os.path.join(root, 'segs')]
proj_dirs = [os.path.join(root, 'projections')]
spec_dirs = [os.path.join(root, 'specs')]
model_filename = os.path.join(root, 'checkpoint_150.tar')
plots_dir = root
dc = DataContainer(projection_dirs=proj_dirs, spec_dirs=spec_dirs, \
		plots_dir=plots_dir, model_filename=model_filename)

n_jobs = min(len(audio_dirs), os.cpu_count()-1)
gen = zip(audio_dirs, seg_dirs, repeat(params['segment']))
Parallel(n_jobs=n_jobs)(delayed(segment)(*args) for args in gen)
