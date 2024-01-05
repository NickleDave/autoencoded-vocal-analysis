"""
Useful functions for preprocessing.

"""
__date__ = "August 2019 - October 2020"


import numpy as np
import warnings
from scipy.signal import stft
from scipy.interpolate import interp2d

from ava.models.vae import Xfill_value_SHAPE


EPSILON = 1e-12


def get_spec(
		t1,
		t2,
		audio,
		num_freq_bins: int = X_SHAPE[0],
		num_time_bins: int = X_SHAPE[1],
		nperseg: int = 1024,
		noverlap: int = 512,
		min_freq: float = 30e3,
		max_freq: float = 110e3,
		spec_min_val: float = 2.0,
		spec_max_val: float = 6.0,
		fs=32000,
		target_freqs=None,
		target_times=None,
		mel: bool = False,
		time_stretch: bool = True,
		fill_value= -1 / EPSILON,
		max_dur: float = 0.2,
		remove_dc_offset=True
	):
	"""
	Norm, scale, threshold, stretch, and resize a Short Time Fourier Transform.

	Notes
	-----
	* ``fill_value`` necessary?
	* Look at all references and see what can be simplified.
	* Why is a flag returned?

	Parameters
	----------
	t1 : float
		Onset time.
	t2 : float
		Offset time.
	audio : numpy.ndarray
		Raw audio.
	p : dict
		Parameters. Must include keys: ...
	fs : float
		Samplerate.
	target_freqs : numpy.ndarray or ``None``, optional
		Interpolated frequencies.
	target_times : numpy.ndarray or ``None``, optional
		Intepolated times.
	fill_value : float, optional
		Defaults to ``-1/EPSILON``.
	max_dur : float, optional
		Maximum duration. Defaults to ``None``.
	remove_dc_offset : bool, optional
		Whether to remove any DC offset from the audio. Defaults to ``True``.

	Returns
	-------
	spec : numpy.ndarray
		Spectrogram.
	flag : bool
		``True``
	"""
	if t2 - t1 > max_dur + 1e-4:
		message = "Found segment longer than max_dur: " + str(t2-t1) + \
				"s, max_dur = " + str(max_dur) + "s"
		warnings.warn(message)
	s1, s2 = int(round(t1*fs)), int(round(t2*fs))
	assert s1 < s2, "s1: " + str(s1) + " s2: " + str(s2) + " t1: " + str(t1) + \
			" t2: " + str(t2)
	# Get a spectrogram and define the interpolation object.
	temp = min(len(audio),s2) - max(0,s1)
	if temp < nperseg or s2 <= 0 or s1 >= len(audio):
		return np.zeros((num_freq_bins, num_time_bins)), True
	else:
		temp_audio = audio[max(0,s1):min(len(audio),s2)]
		if remove_dc_offset:
			temp_audio = temp_audio - np.mean(temp_audio)
		f, t, spec = stft(temp_audio, fs=fs, nperseg=nperseg,
				noverlap=noverlap)
	t += max(0,t1)
	spec = np.log(np.abs(spec) + EPSILON)
	interp = interp2d(t, f, spec, copy=False, bounds_error=False, fill_value=fill_value)
	# Define target frequencies.
	if target_freqs is None:
		if mel:
			target_freqs = np.linspace(_mel(min_freq), _mel(max_freq), num_freq_bins)
			target_freqs = _inv_mel(target_freqs)
		else:
			target_freqs = np.linspace(min_freq, max_freq, \
					p['num_freq_bins'])
	# Define target times.
	if target_times is None:
		duration = t2 - t1
		if time_stretch:
			duration = np.sqrt(duration * max_dur) # stretched duration
		shoulder = 0.5 * (max_dur - duration)
		target_times = np.linspace(t1-shoulder, t2+shoulder, num_time_bins)
	# Then interpolate.
	interp_spec = interp(target_times, target_freqs, assume_sorted=True)
	spec = interp_spec
	# Normalize.
	spec -= spec_min_val
	spec /= (spec_max_val - spec_min_val)
	spec = np.clip(spec, 0.0, 1.0)
	return spec, True


def _mel(a):
	"""https://en.wikipedia.org/wiki/Mel-frequency_cepstrum"""
	return 1127 * np.log(1 + a / 700)


def _inv_mel(a):
	"""https://en.wikipedia.org/wiki/Mel-frequency_cepstrum"""
	return 700 * (np.exp(a / 1127) - 1)



if __name__ == '__main__':
	pass


###
