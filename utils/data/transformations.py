import numpy as np
import librosa

class ToMelSpectrogram(object):

    def __init__(self, sampling_rate=22050, nr_mels=256, nr_fft=2048, hop_size=512, power=2, fmin=40, fmax=11025):
        self.sampling_rate = sampling_rate
        self.nr_mels = nr_mels
        self.power = power
        self.fmin = fmin
        self.fmax = fmax
        self.nr_fft = nr_fft
        self.hop_size = hop_size

        self.melW = librosa.filters.mel(
            sr=sampling_rate,
            n_fft=nr_fft,
            n_mels=nr_mels,
            fmin=fmin,
            fmax=fmax
        ).T

    def __call__(self, x):

        samples = x['x']

        if len(samples.shape) == 1:
            samples = samples.reshape((1, -1))

        channels = []

        for sample in samples:

            stft_matrix = librosa.core.stft(
                y=sample,
                n_fft=self.nr_fft,
                hop_length=self.hop_size,
                window=np.hanning(self.nr_fft),
                center=True,
                dtype=np.complex64,
                pad_mode='reflect'
            ).T

            mel_spectrogram = np.dot(np.abs(stft_matrix) ** self.power, self.melW)

            channels.append(mel_spectrogram.T)

        x['x'] = np.stack(channels, axis=0)
        return x


class ToDB(object):

    def __call__(self, sample):

        sample['x'] = librosa.core.power_to_db(
            sample['x'],
            ref=1.0,
            amin=1e-10,
            top_db=None
        )

        return sample
