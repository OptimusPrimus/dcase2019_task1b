import numpy as np
import pathlib
import librosa
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data.dataset import Dataset
import utils.common
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import torchvision


class SpecDCASE20191b(Dataset):

    def __init__(self, **params):

        self.params = params
        # augment
        self.augment = False
        self.transform = torchvision.transforms.Compose([])
        # id
        self.cache_transformations = params['cache_transformations']
        self.id = [self.__class__.__name__]
        for i, t in enumerate(self.cache_transformations):
            self.id.append(t['class'])
            for p in sorted(t.get('params', {}).keys()):
                self.id += [p, str(t['params'][p])]
            self.cache_transformations[i] = utils.common.load_class(t['class'], **t.get('params', {}))
        self.id = "_".join(self.id)
        # folder
        self.folder_raw_audio = (pathlib.Path(params['folder_raw_audio']) / 'dcase20191b').expanduser()
        self.folder_cache = (pathlib.Path(params['folder_transformed_audio']) / self.id).expanduser()
        #
        self.label_dict, self.class_encoder, self.device_encoder = self.load_label_dict()
        self.folds, self.leader_board, self.submission = self.load_folds()

        self.fold = -1
        self.phase = None
        self.files = None
        self.labels = None
        self.set_phase(0, 'train')

        if not self.is_cached():
            self.cache_data_set()

        self.input_shape = self.get_input_shape()
        self.output_shape = self.get_output_shape()

    def __getitem__(self, idx):

        if self.phase in ['train', 'val']:
            y, dev = self.label_dict[self.files[idx]]
            y = int(y)
        else:
            y = 0
            dev = 0

        sample = {
            'spectrogram': np.load(self.folder_cache / (self.files[idx] + '.npy'))[:, :, :self.input_shape[-1]],
            'label': y,
            'device': dev,
            'idx': idx
        }

        if self.augment:
            sample['dataset'] = self
            sample = self.transform(sample)
            sample.pop('dataset', None)

        return sample

    def get_parallel_set(self):
        return ParallelDataSet(self.files, self.labels, self, self.phase)

    def load_label_dict(self):
        meta = np.loadtxt(self.folder_raw_audio / 'meta.csv', skiprows=1, dtype=np.object)[:, [0, 1, 3]]
        class_encoder = LabelEncoder()
        meta[:, 1] = class_encoder.fit_transform(meta[:, 1])
        device_encoder = LabelEncoder()
        meta[:, 2] = device_encoder.fit_transform(meta[:, 2])
        label_dict = {}
        for i, sample in enumerate(meta):
            name = sample[0].split('.')[0]
            label_dict[name] = sample[1:].astype(np.int64)
        for path in (self.folder_raw_audio / 'test').iterdir():
            name = 'test/' + path.name.split('.')[0]
            label_dict[name] = None
        for path in (self.folder_raw_audio / 'submission').iterdir():
            name = 'submission/' + path.name.split('.')[0]
            label_dict[name] = None
        return label_dict, class_encoder, device_encoder

    def load_folds(self):

        train = list(set([s.split('.')[0] for s in np.loadtxt(
                    self.folder_raw_audio / 'training_setup' / 'fold1_train.csv',
                    skiprows=1,
                    dtype=np.object)[:, 0]
                     ]))

        val = list(set([s.split('.')[0] for s in np.loadtxt(
                    self.folder_raw_audio / 'training_setup' / 'fold1_evaluate.csv',
                    skiprows=1,
                    dtype=np.object)[:, 0]
                     ]))

        return ([
                    [train, val],
                ], \
                sorted('test/' + path.name.split('.')[0] for path in (self.folder_raw_audio / 'test').iterdir()),
                sorted('submission/' + path.name.split('.')[0] for path in (self.folder_raw_audio / 'submission').iterdir())
        )

    def __len__(self):
        return len(self.files)

    def set_phase(self, fold, phase):
        assert phase in ['train', 'val', 'test', 'submission']

        self.phase = phase
        if phase is 'train':
            self.augment = True
            assert fold in range(len(self.folds))
            self.files = self.folds[fold][0]
        elif phase is 'val':
            self.augment = False
            assert fold in range(len(self.folds))
            self.files = self.folds[fold][1]
        elif phase is 'test':
            self.augment = False
            self.files = self.leader_board
        elif phase is 'submission':
            self.augment = False
            self.files = self.submission
        else:
            raise AttributeError

        self.labels = np.array([self.label_dict[f] for f in self.files])

    def is_cached(self):
        return (self.folder_cache.exists() and
                (self.folder_cache / 'test').exists() and
                (self.folder_cache / 'audio').exists() and
                (self.folder_cache / 'submission').exists() and
                len(list((self.folder_cache / 'test').glob('*.npy'))) == len(list((self.folder_raw_audio / 'test').glob('*.wav'))) and
                len(list((self.folder_cache / 'audio').glob('*.npy'))) == len(list((self.folder_raw_audio / 'audio').glob('*.wav'))) and
                len(list((self.folder_cache / 'submission').glob('*.npy'))) == len(list((self.folder_raw_audio / 'submission').glob('*.wav'))))

    def cache_data_set(self):
        (self.folder_cache / 'audio').mkdir(parents=True, exist_ok=True)
        (self.folder_cache / 'test').mkdir(parents=True, exist_ok=True)
        (self.folder_cache / 'submission').mkdir(parents=True, exist_ok=True)

        def f(file):

            if (self.folder_cache / (file + '.npy')).exists():
                return

            sample, _ = librosa.load(
                self.folder_raw_audio / (file + '.wav'),
                sr=self.params['sampling_rate'],
                mono=self.params['mono']
            )

            if not self.params['mono'] and len(sample) != 2:
                sample = np.array([sample, sample])

            sample = {
                'x': sample,
                'device': file.split('-')[-1]

            }
            for t in self.cache_transformations:
                sample = t(sample)

            sample = sample['x']

            np.save(
                self.folder_cache / (file + '.npy'),
                sample.astype(np.float32)
            )

        with ThreadPoolExecutor(max_workers=4) as executor:
            future = executor.map(f, list(self.label_dict))
            for _ in tqdm(future, total=len(list(self.label_dict))):
                pass

    def get_input_shape(self):
        if not hasattr(self, 'input_shape'):
            return np.load(self.folder_cache / (self.files[0] + '.npy')).shape
        return self.input_shape

    def get_output_shape(self):
        return len(self.class_encoder.classes_)


class ParallelDataSet(Dataset):

    def __init__(self, files, labels, data_set, phase, alpha=0.2, cache=True):
        self.data_set = data_set
        label_dict = {}

        for i, file in enumerate(files):
            name = '-'.join(file.split('-')[:-1])
            if label_dict.get(name, None):
                label_dict[name] = (labels[i], label_dict[name][1] + 1)
            else:
                label_dict[name] = (labels[i], 1)

        reduced_label_dict = {}
        for k in label_dict:
            if label_dict[k][1] >= 2:
                reduced_label_dict[k] = label_dict[k][0]

        self.files = list(reduced_label_dict.keys())
        self.labels = [reduced_label_dict[f] for f in self.files]
        self.folder_cache = data_set.folder_cache
        self.phase = phase
        self.alpha = alpha

        self.cache = cache

        if self.cache:
            self.files_ = [
                [
                    np.load(self.folder_cache / (file + '-a.npy')),
                    np.load(self.folder_cache / (file + '-b.npy')),
                    np.load(self.folder_cache / (file + '-c.npy'))
                ]

                for file in self.files
            ]


    @staticmethod
    def rotate_sample(x, i):
        length = x.shape[2]
        x_ = np.empty_like(x)
        x_[:, :, :i] = x[:, :, length - i:]
        x_[:, :, i:] = x[:, :, :length - i]
        return x_

    def __getitem__(self, idx):

        if self.cache:
            a = self.files_[idx][0]
            b = self.files_[idx][1]
            c = self.files_[idx][2]
        else:
            a = np.load(self.folder_cache / (self.files[idx] + '-a.npy'))
            b = np.load(self.folder_cache / (self.files[idx] + '-b.npy'))
            c = np.load(self.folder_cache / (self.files[idx] + '-c.npy'))

        if self.phase == 'train':
            # rotate original
            i = np.random.randint(0, self.data_set.input_shape[-1])
            a = self.rotate_sample(a, i)[:, :, :self.data_set.input_shape[-1]]
            b = self.rotate_sample(b, i)[:, :, :self.data_set.input_shape[-1]]
            c = self.rotate_sample(c, i)[:, :, :self.data_set.input_shape[-1]]

            # load another exaple
            idx = np.random.randint(0, len(self))
            if self.cache:
                a_ = self.files_[idx][0]
                b_ = self.files_[idx][1]
                c_ = self.files_[idx][2]
            else:
                a_ = np.load(self.folder_cache / (self.files[idx] + '-a.npy'))
                b_ = np.load(self.folder_cache / (self.files[idx] + '-b.npy'))
                c_ = np.load(self.folder_cache / (self.files[idx] + '-c.npy'))
            # rotate other
            i = np.random.randint(0, self.data_set.input_shape[-1])
            a_ = self.rotate_sample(a_, i)[:, :, :self.data_set.input_shape[-1]]
            b_ = self.rotate_sample(b_, i)[:, :, :self.data_set.input_shape[-1]]
            c_ = self.rotate_sample(c_, i)[:, :, :self.data_set.input_shape[-1]]

            # mix
            w = np.random.beta(self.alpha, self.alpha)
            w = w if w > 0.5 else 1 - w
            a = w * a + (1.0 - w) * a_
            b = w * b + (1.0 - w) * b_
            c = w * c + (1.0 - w) * c_

        return {
            'spectrogram_a': a[:, :, :self.data_set.input_shape[-1]],
            'spectrogram_b': b[:, :, :self.data_set.input_shape[-1]],
            'spectrogram_c': c[:, :, :self.data_set.input_shape[-1]]
        }

    def __len__(self):
        return len(self.files)
