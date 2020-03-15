import argparse
from sys import float_info
from math import ceil
import numpy as np
from numpy.fft import fft, ifft
from scipy.io import wavfile
import matplotlib.pyplot as plt


def stretch(x, alpha, window, num_channels, synthesis_hop_factor):
    """
    Perform time stretch of a factor alpha to signal x
    x: input signal, alpha: time-stretch factor,
    num_channels: ?, synthesis_hop_factor: ?,
    returns: ?
    """

    synthesis_hop_size = ceil(num_channels / synthesis_hop_factor)
    analysis_hop_size = ceil(synthesis_hop_size / alpha)

    # TODO: Should be able to completely reconstruct input signal if alpha == 1
    x = np.pad(x, (synthesis_hop_size,))
    y = np.zeros(max(x.size, ceil(x.size * alpha + x.size / analysis_hop_size * alpha)))

    ys_old = float_info.epsilon
    analysis_hop = synthesis_hop = 0

    while analysis_hop <= x.size - (synthesis_hop_size + num_channels):
        # Spectra of two consecutive windows
        xs = fft(window * x[analysis_hop:analysis_hop + num_channels])
        xt = fft(window * x[analysis_hop + synthesis_hop_size:analysis_hop + synthesis_hop_size + num_channels])

        # IFFT and overlap and add
        ys = xt * (ys_old / xs) / abs(ys_old / xs)
        ys_old = ys

        y_cur = y[synthesis_hop:synthesis_hop + num_channels]
        y_cur = np.add(y_cur, window * np.real_if_close(ifft(ys)), out=y_cur, casting='unsafe')

        analysis_hop += analysis_hop_size
        synthesis_hop += synthesis_hop_size

    # TODO: AM scaling due to window sliding
    return y[synthesis_hop_size: ceil(x.size * alpha)]


def sin_signal(fs, duration, f0):
    """
    Generate a sinusoidal signal
    fs: sample frequency, duration: signal duration, f0: frequency
    returns: sinusoid of frequency f0 and length duration*fs
    """
    return np.sin(2 * np.pi * f0 * np.linspace(0, duration, int(duration * fs), endpoint=False))


def normalize(x):
    """
        Normalize signal from (min(x), max(x)) to (-1, 1)
        see: https://github.com/WeAreROLI/JUCE/blob/master/modules/juce_core/maths/juce_MathsFunctions.h#L127
    """
    return -1 + 2 * (x - x.min()) / x.ptp()


if __name__ == '__main__':
    windows = {'bartlett': np.bartlett, 'blackman': np.blackman, 'hamming': np.hamming, 'hanning': np.hanning,
               'kaiser': np.kaiser}

    parser = argparse.ArgumentParser(description='Phase Vocoder')
    parser.add_argument('--input_filename', type=str, help='Input filename')

    parser.add_argument('--test_duration', type=float, default=1.0, help='Test sin duration')
    parser.add_argument('--test_frequency', type=float, default=440.0, help='Test sin frequency')
    parser.add_argument('--test_sampling_frequency', type=int, default=44100, help='Test sin sampling frequency')

    parser.add_argument('--stretch_factor', type=float, default=2, help='Stretch factor')
    parser.add_argument('--num_channels', type=int, default=1024, help='Number of FFT channels')
    parser.add_argument('--synthesis_hop_factor', type=float, default=4, help='Synthesis hop factor')
    parser.add_argument('--window', choices=windows.keys(), default='hanning', help='Window function')
    parser.add_argument('--window_size', type=int, default=1024, help='Window size')

    parser.add_argument('--generate_figures', action='store_true', help='Should generate figures')

    parser.add_argument('--output_filename', type=str, default='output.wav', help='Output filename')

    args = parser.parse_args()

    (sampling_frequency, input_data), _ = wavfile.read(
        args.input_filename) if args.input_filename else args.test_sampling_frequency, sin_signal(
        args.test_sampling_frequency, args.test_duration, args.test_frequency)

    input_data = normalize(input_data)

    symmetric_window = windows[args.window](args.window_size) + [0]  # symmetric about (size - 1) / 2
    output_data = stretch(input_data,
                          args.stretch_factor,
                          symmetric_window,
                          args.num_channels,
                          args.synthesis_hop_factor)

    output_data = normalize(output_data)
    wavfile.write(args.output_filename, int(sampling_frequency), output_data)

    if args.generate_figures:
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(input_data.size) / sampling_frequency, input_data)
        plt.title('Input')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        plt.subplot(2, 1, 2)
        plt.plot(np.arange(output_data.size) / sampling_frequency, output_data)
        plt.title('Output')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.show()