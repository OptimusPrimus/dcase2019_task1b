import numpy as np


class MixUp:

    def __init__(self, alpha=0.2, rotate=True, mixup_labels=True):
        self.alpha = alpha
        self.rotate = rotate
        self.mixup_labels = mixup_labels

    @staticmethod
    def rotate_sample(x):
        length = x.shape[2]
        i = np.random.randint(0, length)
        x_ = np.empty_like(x)
        x_[:, :, :i] = x[:, :, length - i:]
        x_[:, :, i:] = x[:, :, :length - i]
        return x_

    @staticmethod
    def one_hot_encode(class_, n_classes):
        v = np.zeros(n_classes)
        v[class_] = 1
        return v

    def __call__(self, sample):

        dataset = sample['dataset']
        a = np.random.beta(self.alpha, self.alpha)
        a = a if a > 0.5 else 1 - a

        if self.mixup_labels:
            idx = np.random.choice(len(dataset))
            y_, _ = dataset.label_dict[dataset.files[idx]]
            sample['label'] = a * self.one_hot_encode(sample['label'], dataset.get_output_shape()) + \
                              (1. - a) * self.one_hot_encode(y_, dataset.get_output_shape())
            sample['label'] = sample['label'].astype(np.float32)
        else:
            idx = np.random.choice(np.argwhere(dataset.labels[:, 0] == sample['label']).reshape(-1))

        spectrogram_ = np.load(dataset.folder_cache / (dataset.files[idx] + '.npy'))[:, :, :dataset.input_shape[-1]]

        if self.rotate:
            sample['spectrogram'] = self.rotate_sample(sample['spectrogram'])
            spectrogram_ = self.rotate_sample(spectrogram_)

        sample['spectrogram'] = a * sample['spectrogram'] + (1. - a) * spectrogram_

        return sample