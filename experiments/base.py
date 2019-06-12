import importlib
import torch
import torchvision
from utils.common import load_class
from torchsummary import summary

class Experiment():

    def __init__(self, experiment, _run, _rnd, **config):

        self.experiment = experiment
        self.config = config
        self._run = _run
        self._rnd = _rnd

        training = config['training']

        # number epochs
        self.nr_epochs = training['nr_epochs']
        # batch size
        self.batch_size = training['batch_size']
        # folds
        self.folds = training.get('folds', None)
        # nr_threads
        self.nr_threads = config['resources']['nr_threads']

        # sampling
        self.sampler = training.get('sampler', None)
        if self.sampler:
            self.sampler = load_class(
                self.sampler['class'],
                **self.sampler.get('params', {})
            )

        # load augmentation pipeline
        self.transform = training.get('augment', [])
        if self.transform is None:
            self.transform = []
        for i, augment_module in enumerate(self.transform):
            self.transform[i] = load_class(
                augment_module['class'],
                **augment_module.get('params', {})
            )

        # load metrics
        self.logger = load_class(
            training['logger']['class'],
            _run,
            **training['logger'].get('params', {})
        )

        # load data set
        self.data_set = load_class(
            config['data_set']['class'],
            **config['data_set'].get('params', {})
        )

        self.data_set.transform = torchvision.transforms.Compose(self.transform)

        # load model
        self.model = getattr(importlib.import_module(config['model']['class']), 'get_model')(
            input_shape=self.data_set.get_input_shape(),
            output_shape=self.data_set.get_output_shape(),
            **config['model'].get('params', {})
        )

        # load GPU
        if config['resources'].get('gpus', None):
            gpus = config['resources'].get('gpus')

            self.devices = [
                torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu") for gpu in gpus
            ]

        else:
            self.devices = [torch.device("cpu:0")]

        # load optimizer
        self.optimizer = load_class(
            training['optimizer']['class'],
            self.model.parameters(),
            **training['optimizer'].get('params', {})
        )

        # load learning rate scheduler
        self.learning_rate_scheduler = training.get('learning_rate_scheduler', None)
        if self.learning_rate_scheduler:
            self.learning_rate_scheduler_metric = self.learning_rate_scheduler.get('metric', None)
            self.learning_rate_scheduler_patience = self.learning_rate_scheduler.get('params', {}).get('patience', None)
            self.learning_rate_scheduler_reset = self.learning_rate_scheduler.get('reset', True)
            self.learning_rate_scheduler_min_lr = self.learning_rate_scheduler.get('params', {}).get('min_lr', -1)
            self.learning_rate_scheduler = load_class(
                self.learning_rate_scheduler['class'],
                self.optimizer,
                **self.learning_rate_scheduler.get('params', {})
            )

        # load loss
        self.loss = load_class(
            training['loss']['class'],
            **training['loss'].get('params', {})
        )

    def run(self):
        raise NotImplementedError
