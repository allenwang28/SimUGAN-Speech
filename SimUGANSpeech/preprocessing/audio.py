# -*- coding: utf-8 *-* 
"""Audio Processing Module

This module provides anything that might be used for audio processing. 

Notes:
    A lot of the implementation is derived from
    - http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
    - https://timsainb.github.io/spectrograms-mfccs-and-inversion-in-python.html

Todo:
    * Make lowcut/highcut normalized by unit
    * sphinx.ext.todo
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
import copy
import soundfile as sf
import os

DEFAULT_BPB_ORDER = 5
DEFAULT_PRE_EMPHASIS = 0.97 

DEFAULT_REAL = False
DEFAULT_ONE_SIDED = True
DEFAULT_SPECTRO_LOG = True
DEFAULT_SPECTRO_THRESH = 5

# TODO - Make lowcut/highcut normalized by unit
DEFAULT_LOWCUT = 300
DEFAULT_HIGHCUT = 3000

DEFAULT_NFFT = 256
DEFAULT_NUM_FILTERS = 40

DEFAULT_FRAME_SIZE_MS = 25
DEFAULT_FRAME_STRIDE_MS = 10

DEFAULT_INVERT_ITER = 15

DEFAULT_WINDOW_FUNCTION = None
DEFAULT_MAX_TIME = None


class AudioParams(object):
    """Helper class to specify spectrogram parameters"""

    def __init__(self):
        """Initialization to only default values."""
        self.bpb_order = DEFAULT_BPB_ORDER
        self.pre_emphasis = DEFAULT_PRE_EMPHASIS
        self.real = DEFAULT_REAL
        self.one_sided = DEFAULT_ONE_SIDED
        self.spectro_log = DEFAULT_SPECTRO_LOG
        self.spectro_thresh = DEFAULT_SPECTRO_THRESH
        self.lowcut = DEFAULT_LOWCUT
        self.highcut = DEFAULT_HIGHCUT
        self.nfft = DEFAULT_NFFT
        self.num_filters = DEFAULT_NUM_FILTERS
        self.frame_size_in_ms = DEFAULT_FRAME_SIZE_MS
        self.frame_stride_in_ms = DEFAULT_FRAME_STRIDE_MS
        self.invert_iter = DEFAULT_INVERT_ITER
        self.window_function = DEFAULT_WINDOW_FUNCTION
        self.max_time = DEFAULT_MAX_TIME 

def get_spectrogram_from_path(file_path,
                              highcut=DEFAULT_HIGHCUT,
                              lowcut=DEFAULT_LOWCUT,
                              log=DEFAULT_SPECTRO_LOG,
                              thresh=DEFAULT_SPECTRO_THRESH,
                              frame_size_in_ms=DEFAULT_FRAME_SIZE_MS,
                              frame_stride_in_ms=DEFAULT_FRAME_STRIDE_MS,
                              real=DEFAULT_REAL):
    """Get the spectrogram from a file.

    Args:
        file_path (str): Path to file
        log (:obj:`bool`, optional): Whether to apply log transform
        thresh (:obj:`int`, optional): Threshold minimum power for log spectrogram
        frame_size_in_ms (:obj:`int`, optional): Size for fast fourier transform
        frame_stride_in_ms (:obj:`int`, optional): Step size for the spectrogram
        real (:obj:`bool`, optional): Whether or not we are dealing with only real numbers

    Returns:
        np.array: Spectrogram
    """
    signal, samplerate = get_file_data(file_path)
    signal = butter_bandpass_filter(signal, samplerate, lowcut, highcut)

    return stft_spectrogram(signal.astype('float64'),
                            samplerate,
                            frame_size_in_ms=frame_size_in_ms,
                            frame_stride_in_ms=frame_stride_in_ms,
                            log=log,
                            thresh=thresh)


def get_file_data(audio_file_path, max_time=DEFAULT_MAX_TIME):
    """Get the signal and sample rate from a given audio file

    Gets the signal as an np.array and sample rate given an audio file path
    using soundfile.

    Args:
        audio_file_path (str): The path to the audio file.
            Supported file types:
                - flac
        max_time (:obj:`int`, optional): Max time (s) for the audio file
            Defaults to None

    Returns:
        np.array : the data of the audio file 
        int      : the sample rate of the audio file

    """
    with open(audio_file_path, 'rb') as f:
        signal, samplerate = sf.read(f)
    if max_time:
        if np.shape(signal)[0]/float(samplerate) > max_time:
            signal = signal[0:samplerate*max_time]
    return (signal, samplerate)


def butter_bandpass_filter(signal, 
                           samplerate, 
                           lowcut=DEFAULT_LOWCUT, 
                           highcut=DEFAULT_HIGHCUT, 
                           order=DEFAULT_BPB_ORDER):
    """Apply a butter bandpass filter to the data.

    Args:
        signal (np.array): The signal from the audio file
        samplerate (int): Sample rate of the audio file
        lowcut (:obj:`int`, optional): Low frequency cut-off (Hz)
        highcut (:obj:`int`, optional): High frequency cut-off (Hz)
        order (:obj:`int`, optional): Order of the butter bandpass

    Returns:
        np.array : Original signal with applied bandpass filter
    """
    nyq = 0.5 * samplerate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, signal)
    return y


def extract_frames_from_signal(signal,
                               samplerate,
                               frame_size_in_ms=DEFAULT_FRAME_SIZE_MS,
                               frame_stride_in_ms=DEFAULT_FRAME_STRIDE_MS,
                               window_function=DEFAULT_WINDOW_FUNCTION):
    """Extracts window frames from a provided signal.

    Args:
        signal (np.array): the input signal.
        samplerate (int): the sample rate.
        frame_size_in_ms (:obj:`int`, optional): frame size in milliseconds.
            Defaults to 25.
        frame_stride_in_ms (:obj:`int`, optional): frame stride in milliseconds. 
            Defaults to 10.
        window_function (:obj:`function`, optional): window function
            Defaults to None. For some options, refer to:
            https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.window.html

    Returns:
        np.array : The matrix of all extracted windows.

    """
    frame_size = frame_size_in_ms / 1000.
    frame_stride = frame_stride_in_ms / 1000.

    frame_length = frame_size * samplerate
    frame_step = frame_stride * samplerate

    signal_length = len(signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))

    padding_length = (frame_length - signal_length) % frame_length
    padding = np.zeros(padding_length)

    # Append the padded signal
    signal = np.hstack((signal, padding))

    # Make sure that we have at least 1 frame
    num_frames = int(np.ceil(np.abs(len(signal) - frame_length) / frame_step))

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = signal[indices.astype(np.int32, copy=False)] 

    if window_function:
        frames = frames * window_function(frame_length)

    return frames
    

def stft(windowed_signal,
         real=DEFAULT_REAL,
         compute_onesided=DEFAULT_ONE_SIDED):
    """Compute the STFT of a windowed signal

    Compute the short-time Fourier transform for 1D real valued input signal

    Args:
        windowed_signal (np.array): windowed input signal
        real (:obj:`boolean`, optional): whether or not we're dealing with only real numbers
        compute_onesided (:obj:`boolean`, optional): whether to compute only one side 
    
    Returns:
        np.array: STFT of the signal
    """
    if real:
        local_fft = np.fft.rfft
        cut = -1
    else:
        local_fft = np.fft.fft
        cut = None
    if compute_onesided:
        cut = windowed_signal.shape[1] // 2

    size = windowed_signal.shape[1]
    win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))
    windowed_signal = windowed_signal * win[None]
    windowed_signal = local_fft(windowed_signal)[:, :cut]
    return windowed_signal


def apply_pre_emphasis_to_signal(signal, pre_emphasis=DEFAULT_PRE_EMPHASIS):
    """Applies pre-emphasis to a signal

    A pre-emphasis filter amplifies high frequencies in a signal.
    Its equation:

        y(t) = x(t) - a * x(t-1)

    where x is the signal and a is the pre_emphasis coefficient
    

    Args:
        signal (np.array): the input signal.
        pre_emphasis (:obj:`int`, optional): pre_emphasis coefficient.
            Typically used values: 0.95, 0.97. Defaults to 0.97

    Returns:
        np.array : The signal with applied pre-emphasis filter

    """
    return np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])  


def stft_spectrogram(signal,
                     samplerate,
                     log=DEFAULT_SPECTRO_LOG,
                     thresh=DEFAULT_SPECTRO_THRESH,
                     frame_size_in_ms=DEFAULT_FRAME_SIZE_MS,
                     frame_stride_in_ms=DEFAULT_FRAME_STRIDE_MS,
                     real=DEFAULT_REAL):
    """Generate a spectrogram from a signal using stft

    Args:
        signal (np.array): Input signal
        samplerate (int): Sample rate for the input signal
        log (:obj:`bool`, optional): Whether to apply log transform
        thresh (:obj:`int`, optional): Threshold minimum power for log spectrogram
        frame_size_in_ms (:obj:`int`, optional): Size for fast fourier transform
        frame_stride_in_ms (:obj:`int`, optional): Step size for the spectrogram
        real (:obj:`bool`, optional): Whether or not we are dealing with only real numbers

    Returns:
        np.array: Spectrogram
    """
    windowed_signal = extract_frames_from_signal(signal,
                                                 samplerate,
                                                 frame_size_in_ms,
                                                 frame_stride_in_ms) 

    spectrogram = np.abs(stft(windowed_signal,
                              real=real,
                              compute_onesided=True))

    if log:
        spectrogram /= spectrogram.max() # First normalize the volume
        spectrogram = np.log10(spectrogram + np.finfo(float).eps) # Apply log-transform
        spectrogram[spectrogram < -thresh] = - thresh
    else:
        spectrogram[spectrogram < thresh] = thresh

    return spectrogram


def xcorr_offset(x1, x2):
    """Calculate the cross correlation"""
    x1 = x1 - x1.mean()
    x2 = x2 - x2.mean()
    frame_size = len(x2)
    half = frame_size // 2

    corrs = np.convolve(x1.astype('float32'), x2[::-1].astype('float32'))
    corrs[:half] = -1E30
    corrs[-half:] = -1E30
    offset = corrs.argmax() - len(x1)
    return offset


def invert_util(spectrogram, step, calculate_offset, set_zero_phase):
    """Utility function for inversion.

    Code used in calculating one iteration for inversion for spectrogram.

    Args:
        spectrogram (np.array): The input spectrogram
        step (int): Step size (note, this is not the in_ms value)
        calculate_offset (bool): Whether to calculate the offset
        set_zero_phase (bool): Whether to set zero phase

    Returns:
        np.array: An estimate of the signal
    """
    num_frames = spectrogram.shape[0]
    size = int(spectrogram.shape[1] // 2)
    signal = np.zeros((int(spectrogram.shape[0] * step + size))).astype('float64')
    total_windowing_sum = np.zeros((int(spectrogram.shape[0] * step + size)))

    win = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))

    est_start = int(size // 2) - 1
    est_end = est_start + size

    for i in range(num_frames):
        signal_start = int(step * i)
        signal_end = signal_start + size

        if set_zero_phase:
            spectral_slice = spectrogram[i].real + 0j
        else:
            spectral_slice = spectrogram[i]

        signal_estimation = np.real(np.fft.ifft(spectral_slice))[::-1]
        if calculate_offset and i > 0:
            offset_size = size - step
            if offset_size <= 0:
                # Large step size >50%
                # The code works best with high overlap.
                # Try with 75% or greater
                offset_size = step
            offset = xcorr_offset(signal[signal_start:signal_start + offset_size],
                                  signal_estimation[est_start:est_start + offset_size])
        else:
            offset = 0
        signal[signal_start:signal_end] += win * signal_estimation[est_start - offset:est_end - offset]
        total_windowing_sum[signal_start:signal_end] += win
    signal = np.real(signal) / (total_windowing_sum + 1E-6)
    return signal


def invert_stft_spectrogram(spectrogram,
                            samplerate,
                            log=DEFAULT_SPECTRO_LOG, 
                            frame_size_in_ms=DEFAULT_FRAME_SIZE_MS,
                            frame_stride_in_ms=DEFAULT_FRAME_STRIDE_MS,
                            n_iterations=DEFAULT_INVERT_ITER):
    """Retrieve the signal from an STFT spectrogram

    Uses the Griffin-Lim Algorithm. Based on MATLAB implementation. See:
         ----------
        D. Griffin and J. Lim. Signal estimation from modified
        short-time Fourier transform. IEEE Trans. Acoust. Speech
        Signal Process., 32(2):236-243, 1984.
        Malcolm Slaney, Daniel Naar and Richard F. Lyon. Auditory
        Model Inversion for Sound Separation. Proc. IEEE-ICASSP,
        Adelaide, 1994, II.77-80.
        Xinglei Zhu, G. Beauregard, L. Wyse. Real-Time Signal
        Estimation from Modified Short-Time Fourier Transform
        Magnitude Spectra. IEEE Transactions on Audio Speech and
        Language Processing, 08/2007.
        ----------

    Args:
        spectrogram (np.array): The input spectrogram
        log (:obj:`bool`, optional): Whether to apply log transform
        frame_size_in_ms (:obj:`int`, optional): Size for fast fourier transform
        frame_stride_in_ms (:obj:`int`, optional): Step size for the spectrogram
        n_iterations (:obj:`int`, optional): Number of iterations to run

    Returns:
        np.array: The signal

    """
    if log:
        spectrogram = np.power(10, spectrogram)

    spectrogram = np.concatenate([spectrogram, spectrogram[:, ::-1]], axis=1)

    frame_stride = frame_stride_in_ms / 1000. 
    frame_step = int(frame_stride * samplerate)

    reg = np.max(spectrogram) / 1E8
    best = copy.deepcopy(spectrogram)
    for i in range(n_iterations):
        set_zero_phase = (i == 0) # Only set zero phase on the first iteration
        signal = invert_util(best, frame_step, True, set_zero_phase)

        windowed_signal = extract_frames_from_signal(signal, samplerate, frame_size_in_ms, frame_stride_in_ms)
        estimation = stft(windowed_signal, compute_onesided=False)
        phase = estimation / np.maximum(reg, np.abs(estimation))
        best = spectrogram * phase[:len(spectrogram)]
    signal = invert_util(best, frame_step, True, False)
    return np.real(signal)


def frequency_to_mel(f):
    """Convert frequency (Hz) to mel-scale"""
    return 2595 * np.log10(1 + (f/2)/700.)


def mel_to_frequency(m):
    """Convert from mel-scale to frequency (Hz)"""
    return 700 * (10**(m / 2595) - 1)
    
