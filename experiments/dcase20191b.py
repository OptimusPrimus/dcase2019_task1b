import torch
import numpy as np
import constants as c
from tqdm import tqdm
from experiments import Experiment
import copy
from utils.common import load_class


class DCASE20191b_Experiment(Experiment):

    def __init__(self, experiment, _run, _rnd, **config):
        super().__init__(experiment, _run, _rnd, **config)

        training = config['training']

        self.batch_size_da = None
        self.lambda_da = None
        self.domain_adaptation = False
        if training.get('domain_adaptation', None):
            self.domain_adaptation = True
            self.lambda_da = training['domain_adaptation'].get('lambda', None)
            self.batch_size_da = training['domain_adaptation'].get('batch_size', None)
            self.model = load_class(
                training['domain_adaptation']['class'],
                self.model,
                **training['domain_adaptation'].get('params', {})
            )

    def run(self):

        self.model = self.model.to(self.devices[0])

        inital_model_state = copy.deepcopy(self.model.state_dict())
        inital_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        inital_learning_rate_scheduler_state = copy.deepcopy(self.learning_rate_scheduler.state_dict())

        results = []

        for f in self.folds:
            # reset stuff
            self.model.load_state_dict(inital_model_state)
            self.optimizer.load_state_dict(inital_optimizer_state)
            self.learning_rate_scheduler.load_state_dict(inital_learning_rate_scheduler_state)
            # train fold
            results.append(self.train_fold(f))

        return np.mean(results)

    def train_fold(self, fold):

        tmp_folder = c.TMP_FOLDER / ''.join(str(self._run.start_time).split(':'))
        tmp_folder.mkdir(parents=True, exist_ok=True)

        id = ''.join(str(self._run.start_time).split(':') + ['_fold_{}'.format(fold)])

        step = 0
        num_bad_epochs = 0

        best_metric = 0

        best_model_state_dict = copy.deepcopy(self.model.state_dict())
        best_optimizer_state_dict = copy.deepcopy(self.optimizer.state_dict())

        # get mean, std for normalization
        mean, std = self.get_mean_std(fold)
        mean = mean.to(self.devices[0])
        std = std.to(self.devices[0])

        # epochs
        for epoch in tqdm(
                range(self.nr_epochs),
                desc='Training...'
        ):
            # train, validation
            phases = ['train', 'val', 'test', 'submission'] if (epoch >= 150 and epoch % 10 == 0) or (
                        epoch + 1 == self.nr_epochs) else ['train', 'val']
            p = 0
            while p < len(phases):

                phase = phases[p]
                p = p + 1

                # set pahse for model
                if phase.startswith('train'):
                    self.model.train()
                else:
                    self.model.eval()

                # load data set
                loader = self.get_data_loader(fold, phase, batch_size=self.batch_size)

                if self.domain_adaptation == True and phase in ['train', 'val']:
                    da_loader = self.get_parallel_data_loader(fold, phase, batch_size=self.batch_size_da)
                else:
                    da_loader = None

                # train/ val one epoch
                results = []
                for i, batch in enumerate(loader):
                    results.append(
                        self.step(
                            batch,
                            next(da_loader) if da_loader and phase in ['train', 'val'] else None,
                            mean,
                            std,
                            self.model,
                            self.optimizer,
                            self.loss,
                            self.devices[0],
                            phase
                        )
                    )

                if phase in ['test', 'submission', 'val'] and len(phases) >= 4:

                    y_pred = np.concatenate(tuple(d['y_pred'] for d in results), axis=0)
                    if p >= 5:
                        # final epoch
                        np.save(
                            tmp_folder / (id + '_logits_{}_best.npy'.format(phase)),
                            y_pred
                        )
                    else:
                        np.save(
                            tmp_folder / (id + '_logits_{}_{}.npy'.format(phase, epoch)),
                            y_pred
                        )

                    if epoch + 1 == self.nr_epochs and p == 4:
                        phases.append('test')
                        phases.append('submission')
                        self.model.load_state_dict(best_model_state_dict)
                        self.optimizer.load_state_dict(best_optimizer_state_dict)

                # increase number of steps
                if phase == 'train':
                    step += len(self.data_set)
                    _ = self.logger(results, fold, phase, step)

                if phase == 'val':
                    if epoch > 100:
                        torch.save(
                            self.model.state_dict(),
                            tmp_folder / '{}_epoch_{}.pt'.format(id, epoch)
                        )
                    metric = self.logger(results, fold, phase, step)
                    if metric > best_metric:
                        best_model_state_dict = copy.deepcopy(self.model.state_dict())
                        best_optimizer_state_dict = copy.deepcopy(self.optimizer.state_dict())
                        torch.save(
                            {
                                'model': best_model_state_dict,
                                'mean': mean,
                                'std': std
                            },
                            tmp_folder / '{}_best.pt'.format(id)
                        )
                        best_metric = metric
                        num_bad_epochs = 0
                    else:
                        num_bad_epochs += 1
                        # reset model, after patience period
                        if self.learning_rate_scheduler and self.learning_rate_scheduler_patience and self.learning_rate_scheduler_reset:
                            cur_lr = self.optimizer.param_groups[0]['lr'] 
                            if num_bad_epochs > self.learning_rate_scheduler_patience and cur_lr > self.learning_rate_scheduler_min_lr:
                                self.model.load_state_dict(best_model_state_dict)
                                # load optimizer & set current learning rate
                                self.optimizer.load_state_dict(best_optimizer_state_dict)
                                for i, param_group in enumerate(self.optimizer.param_groups):
                                    param_group['lr'] = cur_lr
                                num_bad_epochs = 0
                                self._run.log_scalar('lr cur {} '.format(fold), self.optimizer.param_groups[0]['lr'],
                                                     step)

                    # change learning rate
                    if self.learning_rate_scheduler:
                        if self.learning_rate_scheduler_metric:
                            self.learning_rate_scheduler.step(metric)
                        else:
                            self.learning_rate_scheduler.step()
                        self._run.log_scalar('lr next {} '.format(fold), self.optimizer.param_groups[0]['lr'], step)

        return best_metric

    def step(self, batch, batch_da, mean, std, model, optimizer, loss, device, phase):
        batch['spectrogram'] = (batch['spectrogram'].to(device) - mean) / std
        batch['label'] = batch['label'].to(device)

        if batch_da:
            batch_da['spectrogram_a'] = (batch_da['spectrogram_a'].to(device) - mean) / std
            batch_da['spectrogram_b'] = (batch_da['spectrogram_b'].to(device) - mean) / std
            batch_da['spectrogram_c'] = (batch_da['spectrogram_c'].to(device) - mean) / std

        with torch.set_grad_enabled(phase.startswith('train')):
            optimizer.zero_grad()
            outputs = model(batch['spectrogram'])
            loss_clf = loss(outputs, batch['label'])

            if batch_da:
                loss_da = model.forward_da(
                    batch_da['spectrogram_a'],
                    batch_da['spectrogram_b'],
                    batch_da['spectrogram_c']
                ) * self.lambda_da
                loss = loss_clf + loss_da
            else:
                loss = loss_clf

            if phase.startswith('train'):
                loss.backward()
                optimizer.step()

        return {
            'y_pred': outputs.detach().cpu().numpy(),
            'y_true': batch['label'].cpu().numpy(),
            'loss': [loss.item()],
            'loss_clf': [loss_clf.item()],
            'loss_da': [loss_da.item() if batch_da else 0],
            'device': batch['device'].cpu().numpy(),
            'idx': batch['idx'],
        }

    def get_data_loader(self, fold, phase, batch_size=1):
        self.data_set.set_phase(fold, phase)
        return torch.utils.data.DataLoader(
            self.data_set,
            sampler=self.sampler,
            shuffle=True if phase.startswith('train') and not self.sampler else False,
            batch_size=batch_size,
            num_workers=self.nr_threads,
            drop_last=False,
            pin_memory=True
        )

    def get_parallel_data_loader(self, fold, phase, batch_size=1):

        self.data_set.set_phase(fold, phase)

        def infinite_batch(loader):
            while True:
                for batch in loader:
                    yield batch

        loader = torch.utils.data.DataLoader(
            self.data_set.get_parallel_set(),
            sampler=None,
            shuffle=True if phase.startswith('train') else False,
            batch_size=batch_size,
            num_workers=self.nr_threads,
            drop_last=False,
            pin_memory=True
        )

        return infinite_batch(loader)

    def get_mean_std(self, fold):
        loader = self.get_data_loader(fold, "train", batch_size=1)
        self.data_set.augment = False
        mean = None
        for batch in tqdm(loader, desc='Computing mean...'):
            if mean is None:
                mean = torch.mean(batch['spectrogram'], (0, 3), keepdim=True)
            else:
                mean += torch.mean(batch['spectrogram'], (0, 3), keepdim=True)
        mean = mean / len(loader)

        var = None
        for batch in tqdm(loader, desc='Computing std...'):
            if var is None:
                var = torch.mean(torch.pow(batch['spectrogram'] - mean, 2), (0, 3), keepdim=True)
            else:
                var += torch.mean(torch.pow(batch['spectrogram'] - mean, 2), (0, 3), keepdim=True)
        std = torch.sqrt(var / len(loader))

        return mean, std
