import scipy as sp
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score


class DCASE20191b_Logger():

    def __init__(self, _run, ):
        self._run = _run

    def __call__(self, results, fold, phase, step, *args, **kwargs):

        if not phase in ['train', 'test', 'val']:
            raise AttributeError

        if len(results) <= 0:
            return 0

        results_dict = {}
        for k in results[0].keys():
            results_dict[k] = np.concatenate(tuple(d[k] for d in results), axis=0)

        results_dict['y_pred_'] = np.argmax(results_dict['y_pred'], axis=1)

        if len(results_dict['y_true'].shape) == 2:
            results_dict['y_true'] = np.argmax(results_dict['y_true'], axis=1)

        acc_a = accuracy_score(results_dict['y_true'][results_dict['device'] == 0],
                               results_dict['y_pred_'][results_dict['device'] == 0])
        acc_b = accuracy_score(results_dict['y_true'][results_dict['device'] == 1],
                               results_dict['y_pred_'][results_dict['device'] == 1])
        acc_c = accuracy_score(results_dict['y_true'][results_dict['device'] == 2],
                               results_dict['y_pred_'][results_dict['device'] == 2])

        self._run.log_scalar('acc a {} {} '.format(phase, fold), acc_a, step)
        self._run.log_scalar('acc b {} {} '.format(phase, fold), acc_b, step)
        self._run.log_scalar('acc c {} {} '.format(phase, fold), acc_c, step)

        self._run.log_scalar('loss {} {}'.format(phase, fold), np.mean(results_dict['loss']), step)
        self._run.log_scalar('loss_clf {} {}'.format(phase, fold, fold), np.mean(results_dict['loss_clf']), step)
        self._run.log_scalar('loss_da {} {} '.format(phase, fold), np.mean(results_dict['loss_da']), step)

        self._run.log_scalar('acc bc {} {}'.format(phase, fold), np.mean([acc_b, acc_c]), step)

        return np.mean([acc_b, acc_c])